#include "aetheronus/meshing.hpp"
#include "aetheronus/marching_cubes_tables.hpp"
#include "aetheronus/planet_scale.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <limits>
#include <map>
#include <vector>
#include <sstream>

#if defined(__SSE__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#include <immintrin.h>
#endif

namespace ae {
namespace {

using PerfClock = std::chrono::steady_clock;

double elapsed_ms(PerfClock::time_point begin, PerfClock::time_point end = PerfClock::now()) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

constexpr uint32_t MaxSvoDepth = 16u;
constexpr uint32_t SurfacePromotionRootDepth = 8u;
constexpr uint32_t LocalSurfaceNetMaxResolution = 288u;
constexpr float LocalSurfaceNetRootHaloVoxels = 6.0f;

struct Frame {
    Vec3 tangent;
    Vec3 bitangent;
    Vec3 normal;
};

struct GridSample {
    Vec3 local;
    Vec3 world;
    float density = 0.0f;
};

struct ClipVertex {
    Vec3 local;
};

struct FractureShard {
    std::vector<Vec2> polygon;
    Vec2 center;
    uint32_t chunk_id = 0;
};

struct FractureSeed {
    Vec2 local;
    Vec3 direction;
    uint32_t chunk_id = 0;
    float distance_to_cell = 0.0f;
};

struct FracturedLocalSample {
    Vec2 adjusted;
    float top_radius = 0.0f;
    bool is_internal = false;
};

struct FractureBuildCache {
    std::vector<float> chunk_lifts;
};

struct PositionKey {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
};

struct EdgeKey {
    PositionKey a;
    PositionKey b;
};

struct CorridorPointKey {
    PositionKey position;
    uint32_t fracture_chunk_id = 0;
};

struct CellEdgeKey {
    uint32_t cell_id = 0;
    EdgeKey edge;
};

struct BoundaryEdgeRecord {
    uint32_t cell_id = 0;
    EdgeKey edge;
    Vec3 a;
    Vec3 b;
    uint32_t material_id = 0;
    uint32_t fracture_chunk_id = 0;
    uint32_t count = 0;
};

struct BoundarySegment {
    Vec3 a;
    Vec3 b;
    Vec3 midpoint;
    float sort_value = 0.0f;
    uint32_t material_id = 0;
    uint32_t fracture_chunk_id = 0;
};

struct BoundaryPairKey {
    uint32_t a = 0;
    uint32_t b = 0;
};

struct BoundaryPairChains {
    std::vector<BoundarySegment> a_segments;
    std::vector<BoundarySegment> b_segments;
};

struct BoundaryRun {
    std::vector<Vec3> points;
    Vec3 midpoint;
    float min_sort = 0.0f;
    float max_sort = 0.0f;
};

struct BoundaryPairRuns {
    std::vector<BoundaryRun> a_runs;
    std::vector<BoundaryRun> b_runs;
};

struct BoundaryPairGeometry {
    Vec3 edge_a;
    Vec3 edge_b;
    Vec3 edge_axis;
    float edge_length = 0.0f;
};

struct CorridorPoint {
    Vec3 position;
    float sort_value = 0.0f;
    uint32_t material_id = 0;
    uint32_t fracture_chunk_id = 0;
};

struct JunctionCapPoint {
    Vec3 position;
    Vec3 direction;
    uint32_t cell_id = 0;
    float angle = 0.0f;
};

struct SurfaceNetEdgeKey {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t axis = 0;
};

struct PreparedVoxelDig {
    Vec3 center_mesh;
    float radius_mesh = 0.0f;
    float radius_with_leaf_mesh = 0.0f;
};

bool dig_target_matches(const VoxelDigEdit& dig, VoxelDigTarget target) {
    return dig.target == target;
}

struct LocalSurfaceNetPatch {
    Vec3 center_mesh;
    Vec3 dig_center_mesh;
    float dig_radius_mesh = 0.0f;
    float suppress_radius_mesh = 0.0f;
    float extraction_radius_mesh = 0.0f;
    uint32_t depth = MaxSvoDepth;
    bool replace_surface = true;
    std::vector<VoxelKey> promoted_depth8_keys;
    Vec3 replacement_min;
    Vec3 replacement_max;
};

struct LocalSurfaceNetGrid {
    uint32_t depth = MaxSvoDepth;
    uint32_t resolution = 4;
    float voxel_size = 0.0f;
    float radius = 0.0f;
    Vec3 origin;
};

struct SurfaceNetPlacement {
    Vec3 position;
    Vec3 normal;
};

struct CaveCellCandidate {
    uint32_t cell_id = 0;
    Vec3 center = {};
    float inradius_mesh = 0.0f;
};

struct CaveFeatureAnchor {
    uint32_t cell_id = 0;
    Vec3 direction = {};
    float inradius_mesh = 0.0f;
};

struct CaveSurfaceCubeCandidate {
    VoxelKey key;
    uint8_t inside_mask = 0;
};

struct MortonVoxelKey {
    uint64_t code = 0;
    VoxelKey key;
};

struct MortonSurfaceNetEdgeKey {
    uint64_t code = 0;
    SurfaceNetEdgeKey key;
};

using BoundaryEdgeMap = std::vector<BoundaryEdgeRecord>;
using BoundaryPairMap = std::map<BoundaryPairKey, BoundaryPairChains>;
using BoundaryPairRunMap = std::map<BoundaryPairKey, BoundaryPairRuns>;

float mesh_voxel_bounds_radius(const QuantizedMesh& mesh);
void build_svo_node_at(
    std::vector<SparseVoxelOctreeNode>& nodes,
    uint32_t node_index,
    const std::vector<VoxelKey>& keys,
    uint32_t begin,
    uint32_t end,
    uint32_t depth,
    uint32_t origin_x,
    uint32_t origin_y,
    uint32_t origin_z,
    uint32_t size,
    uint32_t max_depth
);
void count_svo_debug_boxes_recursive(
    const SparseVoxelOctree& svo,
    uint32_t node_index,
    uint32_t draw_depth,
    uint32_t max_boxes,
    uint32_t& count
);

bool operator<(const PositionKey& lhs, const PositionKey& rhs) {
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

bool operator==(const PositionKey& lhs, const PositionKey& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool operator<(const EdgeKey& lhs, const EdgeKey& rhs) {
    if (lhs.a < rhs.a) return true;
    if (rhs.a < lhs.a) return false;
    return lhs.b < rhs.b;
}

bool operator<(const CorridorPointKey& lhs, const CorridorPointKey& rhs) {
    if (lhs.position < rhs.position) return true;
    if (rhs.position < lhs.position) return false;
    return lhs.fracture_chunk_id < rhs.fracture_chunk_id;
}

bool operator<(const CellEdgeKey& lhs, const CellEdgeKey& rhs) {
    if (lhs.cell_id != rhs.cell_id) {
        return lhs.cell_id < rhs.cell_id;
    }
    return lhs.edge < rhs.edge;
}

bool operator<(const BoundaryPairKey& lhs, const BoundaryPairKey& rhs) {
    if (lhs.a != rhs.a) {
        return lhs.a < rhs.a;
    }
    return lhs.b < rhs.b;
}

bool operator<(const SurfaceNetEdgeKey& lhs, const SurfaceNetEdgeKey& rhs) {
    if (lhs.axis != rhs.axis) return lhs.axis < rhs.axis;
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

bool operator==(const SurfaceNetEdgeKey& lhs, const SurfaceNetEdgeKey& rhs) {
    return lhs.axis == rhs.axis && lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

uint64_t surface_net_edge_morton_code(SurfaceNetEdgeKey key) {
    return (morton_encode(key.x, key.y, key.z) << 2u) | static_cast<uint64_t>(key.axis & 3u);
}

bool voxel_key_morton_less(VoxelKey lhs, VoxelKey rhs) {
    const uint64_t lhs_code = morton_encode(lhs.x, lhs.y, lhs.z);
    const uint64_t rhs_code = morton_encode(rhs.x, rhs.y, rhs.z);
    if (lhs_code != rhs_code) return lhs_code < rhs_code;
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

bool contains_voxel_key(const std::vector<VoxelKey>& sorted_keys, VoxelKey key) {
    const auto found = std::lower_bound(sorted_keys.begin(), sorted_keys.end(), key, voxel_key_morton_less);
    return found != sorted_keys.end() && *found == key;
}

void sort_and_unique_voxel_keys(std::vector<VoxelKey>& keys) {
    if (keys.size() < 2u) {
        return;
    }

    std::vector<MortonVoxelKey> coded_keys;
    coded_keys.reserve(keys.size());
    for (VoxelKey key : keys) {
        coded_keys.push_back({morton_encode(key.x, key.y, key.z), key});
    }
    std::sort(coded_keys.begin(), coded_keys.end(), [](const MortonVoxelKey& lhs, const MortonVoxelKey& rhs) {
        if (lhs.code != rhs.code) return lhs.code < rhs.code;
        if (lhs.key.x != rhs.key.x) return lhs.key.x < rhs.key.x;
        if (lhs.key.y != rhs.key.y) return lhs.key.y < rhs.key.y;
        return lhs.key.z < rhs.key.z;
    });

    keys.clear();
    keys.reserve(coded_keys.size());
    VoxelKey previous = coded_keys.front().key;
    keys.push_back(previous);
    for (size_t i = 1; i < coded_keys.size(); ++i) {
        const VoxelKey key = coded_keys[i].key;
        if (!(key == previous)) {
            keys.push_back(key);
            previous = key;
        }
    }
}

void sort_and_unique_surface_net_edge_keys(std::vector<SurfaceNetEdgeKey>& keys) {
    if (keys.size() < 2u) {
        return;
    }

    std::vector<MortonSurfaceNetEdgeKey> coded_keys;
    coded_keys.reserve(keys.size());
    for (SurfaceNetEdgeKey key : keys) {
        coded_keys.push_back({surface_net_edge_morton_code(key), key});
    }
    std::sort(coded_keys.begin(), coded_keys.end(), [](const MortonSurfaceNetEdgeKey& lhs, const MortonSurfaceNetEdgeKey& rhs) {
        if (lhs.code != rhs.code) return lhs.code < rhs.code;
        if (lhs.key.axis != rhs.key.axis) return lhs.key.axis < rhs.key.axis;
        if (lhs.key.x != rhs.key.x) return lhs.key.x < rhs.key.x;
        if (lhs.key.y != rhs.key.y) return lhs.key.y < rhs.key.y;
        return lhs.key.z < rhs.key.z;
    });

    keys.clear();
    keys.reserve(coded_keys.size());
    SurfaceNetEdgeKey previous = coded_keys.front().key;
    keys.push_back(previous);
    for (size_t i = 1; i < coded_keys.size(); ++i) {
        const SurfaceNetEdgeKey key = coded_keys[i].key;
        if (!(key == previous)) {
            keys.push_back(key);
            previous = key;
        }
    }
}

void report_progress(const MarchingCubesConfig& config, double progress, const char* label) {
    if (config.progress_callback) {
        config.progress_callback(std::clamp(progress, 0.0, 1.0), label);
    }
}

void report_index_progress(
    const MarchingCubesConfig& config,
    double progress_begin,
    double progress_end,
    uint64_t processed,
    uint64_t total,
    const char* label
) {
    if (total == 0u) {
        return;
    }
    const double t = static_cast<double>(processed) / static_cast<double>(total);
    report_progress(config, progress_begin + (progress_end - progress_begin) * t, label);
}

void validate_morton_helpers_once() {
#ifndef NDEBUG
    static bool validated = false;
    if (validated) {
        return;
    }
    validated = true;

    constexpr std::array<VoxelKey, 5> samples = {{
        {0u, 0u, 0u},
        {1u, 2u, 3u},
        {255u, 128u, 64u},
        {65535u, 32768u, 12345u},
        {0x1fffffu, 0x155555u, 0x0aaaau},
    }};
    for (VoxelKey sample : samples) {
        uint32_t x = 0;
        uint32_t y = 0;
        uint32_t z = 0;
        morton_decode(morton_encode(sample.x, sample.y, sample.z), x, y, z);
        if (x != sample.x || y != sample.y || z != sample.z) {
            std::abort();
        }
    }
#endif
}

Frame build_frame(Vec3 normal) {
    const Vec3 reference_axis = std::fabs(normal.y) < 0.92f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference_axis, normal));
    const Vec3 bitangent = cross(normal, tangent);
    return {tangent, bitangent, normal};
}

Vec2 operator+(Vec2 a, Vec2 b) {
    return {a.x + b.x, a.y + b.y};
}

Vec2 operator-(Vec2 a, Vec2 b) {
    return {a.x - b.x, a.y - b.y};
}

Vec2 operator*(Vec2 v, float scalar) {
    return {v.x * scalar, v.y * scalar};
}

float dot2(Vec2 a, Vec2 b) {
    return a.x * b.x + a.y * b.y;
}

float length2(Vec2 v) {
    return std::sqrt(dot2(v, v));
}

Vec2 lerp2(Vec2 a, Vec2 b, float t) {
    return a * (1.0f - t) + b * t;
}

float clamp_axis(float value, float minimum, float maximum) {
    return std::max(minimum, std::min(maximum, value));
}

Vec3 clamp_vec3(Vec3 value, Vec3 minimum, Vec3 maximum) {
    return {
        clamp_axis(value.x, minimum.x, maximum.x),
        clamp_axis(value.y, minimum.y, maximum.y),
        clamp_axis(value.z, minimum.z, maximum.z),
    };
}

float fast_rsqrt_positive(float value) {
#if defined(__SSE__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
    const __m128 x = _mm_set_ss(value);
    __m128 y = _mm_rsqrt_ss(x);
    const __m128 half = _mm_set_ss(0.5f);
    const __m128 three_halves = _mm_set_ss(1.5f);
    y = _mm_mul_ss(y, _mm_sub_ss(three_halves, _mm_mul_ss(_mm_mul_ss(half, x), _mm_mul_ss(y, y))));
    return _mm_cvtss_f32(y);
#else
    return 1.0f / std::sqrt(value);
#endif
}

Vec3 fast_normalize_vec3(Vec3 value) {
    const float len_sq = value.x * value.x + value.y * value.y + value.z * value.z;
    if (len_sq <= 0.000000000001f) {
        return {0.0f, 1.0f, 0.0f};
    }
    const float inv_len = fast_rsqrt_positive(len_sq);
    return {value.x * inv_len, value.y * inv_len, value.z * inv_len};
}

float snap(float value, float step) {
    if (step <= 0.0f) {
        return value;
    }
    return std::round(value / step) * step;
}

PositionKey position_key(Vec3 position) {
    constexpr float KeyScale = 1000000.0f;
    return {
        static_cast<int32_t>(std::round(position.x * KeyScale)),
        static_cast<int32_t>(std::round(position.y * KeyScale)),
        static_cast<int32_t>(std::round(position.z * KeyScale)),
    };
}

EdgeKey edge_key(Vec3 a, Vec3 b) {
    PositionKey key_a = position_key(a);
    PositionKey key_b = position_key(b);
    if (key_b < key_a) {
        std::swap(key_a, key_b);
    }
    return {key_a, key_b};
}

Vec3 local_to_world(Vec3 center, const Frame& frame, Vec3 local) {
    return center + frame.tangent * local.x + frame.bitangent * local.y + frame.normal * local.z;
}

float signed_area(const std::vector<Vec2>& polygon) {
    float area = 0.0f;
    for (uint32_t i = 0; i < polygon.size(); ++i) {
        const Vec2 a = polygon[i];
        const Vec2 b = polygon[(i + 1) % polygon.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return area * 0.5f;
}

Vec2 project_to_cell_plane(Vec3 center, const Frame& frame, Vec3 position) {
    const Vec3 offset = position - center;
    return {dot(offset, frame.tangent), dot(offset, frame.bitangent)};
}

std::vector<Vec2> cell_clip_polygon(const GoldbergTopology& topology, const GoldbergCell& cell, Vec3 center, const Frame& frame) {
    std::vector<Vec2> polygon;
    polygon.reserve(cell.corner_indices.size());
    for (uint32_t corner_index : cell.corner_indices) {
        polygon.push_back(project_to_cell_plane(center, frame, topology.vertices[corner_index].position));
    }

    if (signed_area(polygon) < 0.0f) {
        std::reverse(polygon.begin(), polygon.end());
    }
    return polygon;
}

bool inside_clip_edge(Vec2 point, Vec2 edge_a, Vec2 edge_b) {
    const Vec2 edge = {edge_b.x - edge_a.x, edge_b.y - edge_a.y};
    const Vec2 rel = {point.x - edge_a.x, point.y - edge_a.y};
    return edge.x * rel.y - edge.y * rel.x >= -0.000001f;
}

ClipVertex intersect_clip_edge(const ClipVertex& a, const ClipVertex& b, Vec2 edge_a, Vec2 edge_b) {
    const Vec2 p = {a.local.x, a.local.y};
    const Vec2 q = {b.local.x, b.local.y};
    const Vec2 segment = {q.x - p.x, q.y - p.y};
    const Vec2 edge = {edge_b.x - edge_a.x, edge_b.y - edge_a.y};
    const Vec2 rel = {p.x - edge_a.x, p.y - edge_a.y};
    const float denominator = edge.x * segment.y - edge.y * segment.x;

    float t = 0.0f;
    if (std::fabs(denominator) > 0.000001f) {
        t = std::clamp(-(edge.x * rel.y - edge.y * rel.x) / denominator, 0.0f, 1.0f);
    }

    return {lerp(a.local, b.local, t)};
}

std::vector<ClipVertex> clip_to_cell_polygon(const std::array<GridSample, 3>& triangle, const std::vector<Vec2>& clip_polygon) {
    std::vector<ClipVertex> output = {{{triangle[0].local}, {triangle[1].local}, {triangle[2].local}}};

    for (uint32_t edge_index = 0; edge_index < clip_polygon.size(); ++edge_index) {
        const Vec2 edge_a = clip_polygon[edge_index];
        const Vec2 edge_b = clip_polygon[(edge_index + 1) % clip_polygon.size()];
        const std::vector<ClipVertex> input = output;
        output.clear();
        if (input.empty()) {
            break;
        }

        ClipVertex previous = input.back();
        bool previous_inside = inside_clip_edge({previous.local.x, previous.local.y}, edge_a, edge_b);
        for (const ClipVertex& current : input) {
            const bool current_inside = inside_clip_edge({current.local.x, current.local.y}, edge_a, edge_b);
            if (current_inside) {
                if (!previous_inside) {
                    output.push_back(intersect_clip_edge(previous, current, edge_a, edge_b));
                }
                output.push_back(current);
            } else if (previous_inside) {
                output.push_back(intersect_clip_edge(previous, current, edge_a, edge_b));
            }

            previous = current;
            previous_inside = current_inside;
        }
    }

    return output;
}

Vec3 quantize_local(Vec3 local, Vec3 step) {
    return {
        snap(local.x, step.x),
        snap(local.y, step.y),
        snap(local.z, step.z),
    };
}

GridSample interpolate(const GridSample& a, const GridSample& b, Vec3 center, const Frame& frame, Vec3 quantize_step, bool quantize) {
    const float denominator = a.density - b.density;
    float t = 0.5f;
    if (std::fabs(denominator) > 0.000001f) {
        t = std::clamp(a.density / denominator, 0.0f, 1.0f);
    }

    Vec3 local = lerp(a.local, b.local, t);
    if (quantize) {
        local = quantize_local(local, quantize_step);
    }

    const Vec3 world = local_to_world(center, frame, local);
    return {local, world, 0.0f};
}

GridSample interpolate_projected(
    const GridSample& a,
    const GridSample& b,
    Vec3 center,
    const Frame& frame,
    Vec3 quantize_step,
    bool quantize,
    float surface_radius
) {
    GridSample sample = interpolate(a, b, center, frame, quantize_step, quantize);
    sample.world = normalize(sample.world) * surface_radius;
    return sample;
}

float owned_surface_radius(const PointCloud& points, uint32_t cell_id) {
    return cell_id < points.owned_radius_by_cell.size() ? points.owned_radius_by_cell[cell_id] : 1.0f;
}

float cell_tangent_extent(const GoldbergTopology& topology, const GoldbergCell& cell, const Frame& frame) {
    float extent = 0.02f;
    for (uint32_t corner_index : cell.corner_indices) {
        const Vec3 offset = topology.vertices[corner_index].position - cell.center;
        extent = std::max(extent, std::fabs(dot(offset, frame.tangent)));
        extent = std::max(extent, std::fabs(dot(offset, frame.bitangent)));
    }
    return extent * 1.08f;
}

uint32_t nearest_cell_on_sphere(const GoldbergTopology& topology, Vec3 position) {
    uint32_t nearest = 0;
    float best_dot = -2.0f;
    const Vec3 normalized_position = normalize(position);

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const float score = dot(normalized_position, topology.cells[cell_id].center);
        if (score > best_dot) {
            best_dot = score;
            nearest = cell_id;
        }
    }

    return nearest;
}

uint32_t nearest_neighbor_on_sphere(const GoldbergTopology& topology, uint32_t cell_id, Vec3 position) {
    const Vec3 normalized_position = normalize(position);
    const GoldbergCell& cell = topology.cells[cell_id];
    uint32_t nearest = cell.neighbor_indices.empty() ? cell_id : cell.neighbor_indices.front();
    float best_dot = -2.0f;

    for (uint32_t neighbor_id : cell.neighbor_indices) {
        const float score = dot(normalized_position, topology.cells[neighbor_id].center);
        if (score > best_dot) {
            best_dot = score;
            nearest = neighbor_id;
        }
    }

    return nearest;
}

BoundaryPairKey boundary_pair_key(uint32_t a, uint32_t b) {
    if (b < a) {
        std::swap(a, b);
    }
    return {a, b};
}

void record_triangle_edge(
    BoundaryEdgeMap& edges,
    uint32_t cell_id,
    Vec3 a,
    Vec3 b,
    uint32_t material_id,
    uint32_t fracture_chunk_id
) {
    edges.push_back({
        cell_id,
        edge_key(a, b),
        a,
        b,
        material_id,
        fracture_chunk_id,
        1u,
    });
}

void record_triangle_edges(
    BoundaryEdgeMap& edges,
    uint32_t cell_id,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    uint32_t material_id,
    uint32_t fracture_chunk_id
) {
    record_triangle_edge(edges, cell_id, a, b, material_id, fracture_chunk_id);
    record_triangle_edge(edges, cell_id, b, c, material_id, fracture_chunk_id);
    record_triangle_edge(edges, cell_id, c, a, material_id, fracture_chunk_id);
}

void emit_mesh_triangle(
    QuantizedMesh& mesh,
    BoundaryEdgeMap& boundary_edges,
    uint32_t emitting_cell_id,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    uint32_t material_id,
    uint32_t fracture_chunk_id = 0
) {
    const Vec3 normal = normalize(cross(b - a, c - a));
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({a, normal, material_id, emitting_cell_id, fracture_chunk_id});
    mesh.vertices.push_back({b, normal, material_id, emitting_cell_id, fracture_chunk_id});
    mesh.vertices.push_back({c, normal, material_id, emitting_cell_id, fracture_chunk_id});
    mesh.triangle_indices.push_back(base);
    mesh.triangle_indices.push_back(base + 1);
    mesh.triangle_indices.push_back(base + 2);
    mesh.line_indices.push_back(base);
    mesh.line_indices.push_back(base + 1);
    mesh.line_indices.push_back(base + 1);
    mesh.line_indices.push_back(base + 2);
    mesh.line_indices.push_back(base + 2);
    mesh.line_indices.push_back(base);
    ++mesh.triangle_count;
    record_triangle_edges(boundary_edges, emitting_cell_id, a, b, c, material_id, fracture_chunk_id);
}

bool clip_changed(const std::array<GridSample, 3>& triangle, const std::vector<ClipVertex>& clipped) {
    if (clipped.size() != triangle.size()) {
        return true;
    }

    for (uint32_t i = 0; i < triangle.size(); ++i) {
        const Vec3 delta = clipped[i].local - triangle[i].local;
        if (std::fabs(delta.x) > 0.000001f || std::fabs(delta.y) > 0.000001f) {
            return true;
        }
    }

    return false;
}

void append_clipped_triangle(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    BoundaryEdgeMap& boundary_edges,
    uint32_t emitting_cell_id,
    const std::array<GridSample, 3>& triangle,
    Vec3 center,
    const Frame& frame,
    const std::vector<Vec2>& clip_polygon,
    float surface_radius,
    uint32_t material_id
) {
    const std::vector<ClipVertex> clipped = clip_to_cell_polygon(triangle, clip_polygon);
    if (clipped.size() < 3) {
        ++mesh.discarded_clipped_triangle_count;
        return;
    }

    if (clip_changed(triangle, clipped)) {
        ++mesh.clipped_triangle_count;
    }

    const Vec3 fan_origin = normalize(local_to_world(center, frame, clipped.front().local)) * surface_radius;
    for (uint32_t i = 1; i + 1 < clipped.size(); ++i) {
        const Vec3 b = normalize(local_to_world(center, frame, clipped[i].local)) * surface_radius;
        const Vec3 c = normalize(local_to_world(center, frame, clipped[i + 1].local)) * surface_radius;
        const uint32_t owner = nearest_cell_on_sphere(topology, normalize((fan_origin + b + c) / 3.0f));
        if (owner != emitting_cell_id) {
            ++mesh.rejected_triangle_count;
            continue;
        }

        emit_mesh_triangle(mesh, boundary_edges, emitting_cell_id, fan_origin, b, c, material_id);
    }
}

void emit_oriented_mesh_triangle(
    QuantizedMesh& mesh,
    BoundaryEdgeMap& boundary_edges,
    uint32_t emitting_cell_id,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    Vec3 outward,
    uint32_t material_id,
    uint32_t fracture_chunk_id = 0
) {
    if (dot(cross(b - a, c - a), outward) < 0.0f) {
        std::swap(b, c);
    }
    emit_mesh_triangle(mesh, boundary_edges, emitting_cell_id, a, b, c, material_id, fracture_chunk_id);
}

Vec3 subdivided_cell_vertex(Vec3 center, Vec3 edge_a, Vec3 edge_b, uint32_t a_step, uint32_t b_step, uint32_t subdivisions, float radius) {
    const float inv_subdivisions = 1.0f / static_cast<float>(subdivisions);
    const float a_weight = static_cast<float>(a_step) * inv_subdivisions;
    const float b_weight = static_cast<float>(b_step) * inv_subdivisions;
    const float center_weight = 1.0f - a_weight - b_weight;
    return normalize(center * center_weight + edge_a * a_weight + edge_b * b_weight) * radius;
}

Vec2 subdivided_cell_local_vertex(Vec2 edge_a, Vec2 edge_b, uint32_t a_step, uint32_t b_step, uint32_t subdivisions) {
    const float inv_subdivisions = 1.0f / static_cast<float>(subdivisions);
    return edge_a * (static_cast<float>(a_step) * inv_subdivisions) +
           edge_b * (static_cast<float>(b_step) * inv_subdivisions);
}

uint32_t hash_u32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

uint32_t global_fracture_seed_count(const MarchingCubesConfig& config);

float hash_unit(uint32_t value) {
    return static_cast<float>(hash_u32(value) & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

float compute_fracture_chunk_lift(uint32_t chunk_id, const MarchingCubesConfig& config) {
    const float outward_min = std::max(0.0f, std::min(config.fracture_chunk_outward_min, config.fracture_chunk_outward_max));
    const float outward_max = std::max(outward_min, std::max(config.fracture_chunk_outward_min, config.fracture_chunk_outward_max));
    const float outward_t = hash_unit(config.fracture_seed ^ (chunk_id * 0x9e3779b9u) ^ 0x68bc21ebu);
    return config.fracture_seams_use_max_lift ? outward_max : outward_min + (outward_max - outward_min) * outward_t;
}

FractureBuildCache build_fracture_build_cache(const MarchingCubesConfig& config) {
    FractureBuildCache cache;
    if (!config.enable_fractures || !config.connect_fractures_across_cells) {
        return cache;
    }

    const uint32_t seed_count = global_fracture_seed_count(config);
    cache.chunk_lifts.resize(seed_count + 1u, 0.0f);
    for (uint32_t chunk_id = 1; chunk_id <= seed_count; ++chunk_id) {
        cache.chunk_lifts[chunk_id] = compute_fracture_chunk_lift(chunk_id, config);
    }
    return cache;
}

float fracture_chunk_lift(uint32_t chunk_id, const MarchingCubesConfig& config, const FractureBuildCache& cache) {
    if (chunk_id < cache.chunk_lifts.size()) {
        return cache.chunk_lifts[chunk_id];
    }
    return compute_fracture_chunk_lift(chunk_id, config);
}

bool point_in_convex_polygon(Vec2 point, const std::vector<Vec2>& polygon) {
    if (polygon.size() < 3) {
        return false;
    }
    for (uint32_t i = 0; i < polygon.size(); ++i) {
        if (!inside_clip_edge(point, polygon[i], polygon[(i + 1) % polygon.size()])) {
            return false;
        }
    }
    return true;
}

Vec2 polygon_centroid(const std::vector<Vec2>& polygon) {
    if (polygon.empty()) {
        return {};
    }

    float area_sum = 0.0f;
    Vec2 centroid = {};
    for (uint32_t i = 0; i < polygon.size(); ++i) {
        const Vec2 a = polygon[i];
        const Vec2 b = polygon[(i + 1) % polygon.size()];
        const float cross_value = a.x * b.y - b.x * a.y;
        area_sum += cross_value;
        centroid = centroid + (a + b) * cross_value;
    }

    if (std::fabs(area_sum) <= 0.000001f) {
        Vec2 average = {};
        for (Vec2 point : polygon) {
            average = average + point;
        }
        return average * (1.0f / static_cast<float>(polygon.size()));
    }

    return centroid * (1.0f / (3.0f * area_sum));
}

float point_segment_distance(Vec2 point, Vec2 a, Vec2 b) {
    const Vec2 segment = b - a;
    const float segment_length_sq = dot2(segment, segment);
    if (segment_length_sq <= 0.000001f) {
        return length2(point - a);
    }

    const float t = std::clamp(dot2(point - a, segment) / segment_length_sq, 0.0f, 1.0f);
    return length2(point - lerp2(a, b, t));
}

float distance_to_polygon_boundary(Vec2 point, const std::vector<Vec2>& polygon) {
    float best = 1000000.0f;
    for (uint32_t i = 0; i < polygon.size(); ++i) {
        best = std::min(best, point_segment_distance(point, polygon[i], polygon[(i + 1) % polygon.size()]));
    }
    return best;
}

float cell_polygon_inradius(const std::vector<Vec2>& polygon) {
    if (polygon.size() < 3) {
        return 0.0f;
    }
    return distance_to_polygon_boundary(polygon_centroid(polygon), polygon);
}

Vec3 seeded_unit_direction(uint32_t seed, uint32_t index) {
    const uint32_t base = seed ^ (index * 0x9e3779b9u) ^ 0x4cf5ad43u;
    const float z = hash_unit(base) * 2.0f - 1.0f;
    const float theta = hash_unit(base ^ 0x85ebca6bu) * (Pi * 2.0f);
    const float radius = std::sqrt(std::max(0.0f, 1.0f - z * z));
    return {
        std::cos(theta) * radius,
        std::sin(theta) * radius,
        z,
    };
}

std::vector<Vec3> optimize_cave_directions_frank_wolfe(uint32_t seed, uint32_t cave_count) {
    std::vector<Vec3> targets;
    targets.reserve(cave_count);
    std::vector<Vec3> directions;
    directions.reserve(cave_count);
    for (uint32_t i = 0; i < cave_count; ++i) {
        const Vec3 target = seeded_unit_direction(seed, i);
        targets.push_back(target);
        directions.push_back(target);
    }

    constexpr uint32_t Iterations = 48u;
    constexpr float RepulsionWeight = 1.0f;
    constexpr float SeedAttractionWeight = 0.12f;
    constexpr float RepulsionEpsilon = 0.035f;
    for (uint32_t iteration = 0; iteration < Iterations; ++iteration) {
        std::vector<Vec3> gradients(cave_count);
        for (uint32_t i = 0; i < cave_count; ++i) {
            gradients[i] = targets[i] * -SeedAttractionWeight;
        }

        for (uint32_t i = 0; i < cave_count; ++i) {
            for (uint32_t j = i + 1u; j < cave_count; ++j) {
                const float similarity = std::clamp(dot(directions[i], directions[j]), -0.999f, 0.999f);
                const float denominator = std::max(RepulsionEpsilon, 1.0f - similarity + RepulsionEpsilon);
                const float scale = RepulsionWeight / (denominator * denominator);
                gradients[i] = gradients[i] + directions[j] * scale;
                gradients[j] = gradients[j] + directions[i] * scale;
            }
        }

        const float gamma = 2.0f / static_cast<float>(iteration + 2u);
        for (uint32_t i = 0; i < cave_count; ++i) {
            const Vec3 descent = length(gradients[i]) > 0.000001f
                ? normalize(gradients[i]) * -1.0f
                : targets[i];
            directions[i] = directions[i] * (1.0f - gamma) + descent * gamma;
        }
    }

    for (Vec3& direction : directions) {
        direction = normalize(direction);
    }
    return directions;
}

std::vector<uint32_t> select_frank_wolfe_cave_cells(
    const std::vector<CaveCellCandidate>& candidates,
    uint32_t seed,
    uint32_t cave_count
) {
    std::vector<uint32_t> selected_cells;
    if (candidates.empty() || cave_count == 0u) {
        return selected_cells;
    }

    const uint32_t selected_count = std::min(cave_count, static_cast<uint32_t>(candidates.size()));
    selected_cells.reserve(selected_count);
    const std::vector<Vec3> directions = optimize_cave_directions_frank_wolfe(seed, selected_count);
    std::vector<uint8_t> used(candidates.size(), 0u);

    constexpr float TieBreakerWeight = 0.000001f;
    for (uint32_t feature_index = 0; feature_index < selected_count; ++feature_index) {
        const Vec3 direction = directions[feature_index];
        size_t best_index = candidates.size();
        float best_score = -1000000.0f;
        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
            if (used[candidate_index] != 0u) {
                continue;
            }
            const CaveCellCandidate& candidate = candidates[candidate_index];
            const float tie_breaker = hash_unit(
                seed ^
                (candidate.cell_id * 0x7feb352du) ^
                (feature_index * 0x846ca68bu) ^
                0x51ed270bu
            ) * TieBreakerWeight;
            const float score = dot(candidate.center, direction) + tie_breaker;
            if (score > best_score || (std::fabs(score - best_score) <= 0.0000001f && candidate.cell_id < candidates[best_index].cell_id)) {
                best_score = score;
                best_index = candidate_index;
            }
        }

        if (best_index == candidates.size()) {
            break;
        }
        used[best_index] = 1u;
        selected_cells.push_back(candidates[best_index].cell_id);
    }

    return selected_cells;
}

size_t nearest_cave_cell_candidate(const std::vector<CaveCellCandidate>& candidates, Vec3 direction) {
    size_t nearest_index = candidates.size();
    float best_score = -2.0f;
    const Vec3 normalized_direction = normalize(direction);
    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
        const float score = dot(normalized_direction, candidates[candidate_index].center);
        if (score > best_score) {
            best_score = score;
            nearest_index = candidate_index;
        }
    }
    return nearest_index;
}

std::vector<CaveFeatureAnchor> select_anchor_cave_features(
    const std::vector<CaveCellCandidate>& candidates,
    const std::vector<Vec3>& cave_anchor_points,
    uint32_t seed,
    uint32_t cave_count
) {
    std::vector<CaveFeatureAnchor> selected;
    if (candidates.empty() || cave_anchor_points.empty() || cave_count == 0u) {
        return selected;
    }

    const uint32_t anchor_count = static_cast<uint32_t>(cave_anchor_points.size());
    const uint32_t selected_count = std::min(cave_count, anchor_count);
    selected.reserve(selected_count);
    const uint32_t offset = anchor_count == 0u
        ? 0u
        : hash_u32(seed ^ 0x6d2b79f5u) % anchor_count;
    const uint32_t stride = std::max(1u, anchor_count / std::max(1u, selected_count));

    for (uint32_t feature_slot = 0; feature_slot < selected_count; ++feature_slot) {
        const uint32_t anchor_index = (offset + feature_slot * stride) % anchor_count;
        const Vec3 direction = normalize(cave_anchor_points[anchor_index]);
        const size_t candidate_index = nearest_cave_cell_candidate(candidates, direction);
        if (candidate_index < candidates.size()) {
            selected.push_back({
                candidates[candidate_index].cell_id,
                direction,
                candidates[candidate_index].inradius_mesh,
            });
        }
    }

    return selected;
}

std::vector<LocalVoxelFeature> build_local_voxel_features(
    const GoldbergTopology& topology,
    const MarchingCubesConfig& config,
    const std::vector<Vec3>& cave_anchor_points
) {
    std::vector<LocalVoxelFeature> features;
    if (!config.voxel_features.enabled || config.voxel_features.cave_count == 0u) {
        return features;
    }

    std::vector<CaveCellCandidate> candidates;
    candidates.reserve(topology.cells.size());
    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        if (cell.kind != GoldbergCellKind::Hexagon || cell.corner_indices.size() < 3) {
            continue;
        }
        const Frame frame = build_frame(cell.normal);
        const std::vector<Vec2> polygon = cell_clip_polygon(topology, cell, cell.center, frame);
        const float inradius_mesh = cell_polygon_inradius(polygon);
        if (inradius_mesh <= 0.000001f) {
            continue;
        }
        candidates.push_back({
            cell_id,
            normalize(cell.center),
            inradius_mesh,
        });
    }

    std::vector<CaveFeatureAnchor> selected_anchors = select_anchor_cave_features(
        candidates,
        cave_anchor_points,
        config.voxel_features.seed,
        config.voxel_features.cave_count
    );
    if (selected_anchors.empty()) {
        const std::vector<uint32_t> selected_cells = select_frank_wolfe_cave_cells(
            candidates,
            config.voxel_features.seed,
            config.voxel_features.cave_count
        );
        selected_anchors.reserve(selected_cells.size());
        for (uint32_t cell_id : selected_cells) {
            const size_t candidate_index = nearest_cave_cell_candidate(candidates, topology.cells[cell_id].center);
            if (candidate_index < candidates.size()) {
                selected_anchors.push_back({
                    cell_id,
                    candidates[candidate_index].center,
                    candidates[candidate_index].inradius_mesh,
                });
            }
        }
    }

    const uint32_t cave_count = static_cast<uint32_t>(selected_anchors.size());
    features.reserve(cave_count);
    for (uint32_t feature_index = 0; feature_index < cave_count; ++feature_index) {
        const CaveFeatureAnchor& selected_anchor = selected_anchors[feature_index];
        const uint32_t cell_id = selected_anchor.cell_id;
        const Vec3 anchor_direction = normalize(selected_anchor.direction);
        const Frame frame = build_frame(anchor_direction);
        const float inradius_mesh = selected_anchor.inradius_mesh;
        const float requested_entrance_radius = kilometers_to_world_units(config.voxel_features.entrance_radius_km);
        const float entrance_radius_mesh = std::max(
            kilometers_to_world_units(6.0f),
            std::min(requested_entrance_radius, inradius_mesh * 0.42f)
        );
        const uint32_t seed = config.voxel_features.seed ^
            (cell_id * 0x85ebca6bu) ^
            (feature_index * 0xc2b2ae35u);

        LocalVoxelFeature feature;
        feature.kind = VoxelFeatureKind::CaveSystem;
        feature.feature_id = feature_index + 1u;
        feature.owner_cell_id = cell_id;
        feature.center_mesh = anchor_direction;
        feature.normal_mesh = anchor_direction;
        feature.tangent_mesh = frame.tangent;
        feature.bitangent_mesh = frame.bitangent;
        feature.entrance_radius_km = world_units_to_kilometers(entrance_radius_mesh);
        feature.tunnel_radius_km = std::min(config.voxel_features.tunnel_radius_km, feature.entrance_radius_km * 0.82f);
        feature.chamber_radius_km = config.voxel_features.chamber_radius_km;
        feature.depth_km = config.voxel_features.cave_depth_km;
        feature.svo_depth = std::clamp(config.voxel_features.cave_depth, 1u, MaxSvoDepth);
        feature.seed = seed;
        features.push_back(feature);
    }

    return features;
}

std::vector<Vec2> clip_polygon_to_voronoi_half_plane(const std::vector<Vec2>& polygon, Vec2 seed, Vec2 other_seed) {
    std::vector<Vec2> output;
    if (polygon.empty()) {
        return output;
    }

    const Vec2 axis = (other_seed - seed) * 2.0f;
    const float threshold = dot2(other_seed, other_seed) - dot2(seed, seed);
    auto signed_distance = [&](Vec2 point) {
        return dot2(axis, point) - threshold;
    };

    Vec2 previous = polygon.back();
    float previous_distance = signed_distance(previous);
    bool previous_inside = previous_distance <= 0.000001f;
    for (Vec2 current : polygon) {
        const float current_distance = signed_distance(current);
        const bool current_inside = current_distance <= 0.000001f;
        if (current_inside != previous_inside) {
            const float denominator = previous_distance - current_distance;
            const float t = std::fabs(denominator) > 0.000001f ? std::clamp(previous_distance / denominator, 0.0f, 1.0f) : 0.0f;
            output.push_back(lerp2(previous, current, t));
        }
        if (current_inside) {
            output.push_back(current);
        }

        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }

    return output;
}

std::vector<Vec2> clip_polygon_to_spherical_voronoi_half_plane(
    const std::vector<Vec2>& polygon,
    Vec3 cell_center,
    const Frame& frame,
    Vec3 seed_direction,
    Vec3 other_seed_direction
) {
    std::vector<Vec2> output;
    if (polygon.empty()) {
        return output;
    }

    const Vec3 separator = seed_direction - other_seed_direction;
    auto signed_distance = [&](Vec2 point) {
        const Vec3 world = normalize(local_to_world(cell_center, frame, {point.x, point.y, 0.0f}));
        return dot(world, separator);
    };

    auto edge_intersection = [&](Vec2 a, Vec2 b) {
        float lo = 0.0f;
        float hi = 1.0f;
        const bool a_inside = signed_distance(a) >= -0.000001f;
        for (uint32_t i = 0; i < 14; ++i) {
            const float mid = (lo + hi) * 0.5f;
            const Vec2 candidate = lerp2(a, b, mid);
            const bool mid_inside = signed_distance(candidate) >= -0.000001f;
            if (mid_inside == a_inside) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lerp2(a, b, (lo + hi) * 0.5f);
    };

    Vec2 previous = polygon.back();
    float previous_distance = signed_distance(previous);
    bool previous_inside = previous_distance >= -0.000001f;
    for (Vec2 current : polygon) {
        const float current_distance = signed_distance(current);
        const bool current_inside = current_distance >= -0.000001f;
        if (current_inside != previous_inside) {
            output.push_back(edge_intersection(previous, current));
        }
        if (current_inside) {
            output.push_back(current);
        }

        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }

    return output;
}

Vec2 deterministic_point_in_polygon(const std::vector<Vec2>& polygon, uint32_t seed) {
    const Vec2 center = polygon_centroid(polygon);
    Vec2 min_bounds = polygon.front();
    Vec2 max_bounds = polygon.front();
    for (Vec2 point : polygon) {
        min_bounds.x = std::min(min_bounds.x, point.x);
        min_bounds.y = std::min(min_bounds.y, point.y);
        max_bounds.x = std::max(max_bounds.x, point.x);
        max_bounds.y = std::max(max_bounds.y, point.y);
    }

    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        const uint32_t hash_base = seed ^ (attempt * 0x85ebca6bu);
        const Vec2 candidate = {
            min_bounds.x + (max_bounds.x - min_bounds.x) * hash_unit(hash_base),
            min_bounds.y + (max_bounds.y - min_bounds.y) * hash_unit(hash_base ^ 0xa511e9b3u),
        };
        if (point_in_convex_polygon(candidate, polygon)) {
            return lerp2(center, candidate, 0.94f);
        }
    }

    return center;
}

Vec3 global_fracture_seed_position(uint32_t seed_id, uint32_t seed_count, const MarchingCubesConfig& config) {
    const float count = static_cast<float>(std::max(1u, seed_count));
    const float index = static_cast<float>(seed_id);
    const float z = 1.0f - 2.0f * ((index + 0.5f) / count);
    const float radius = std::sqrt(std::max(0.0f, 1.0f - z * z));
    constexpr float GoldenAngle = 2.39996323f;
    const float phase = hash_unit(config.fracture_seed ^ 0x7f4a7c15u) * 2.0f * Pi;
    const float theta = index * GoldenAngle + phase;
    return normalize({std::cos(theta) * radius, z, std::sin(theta) * radius});
}

uint32_t global_fracture_seed_count(const MarchingCubesConfig& config) {
    return std::max(4u, config.global_fracture_seed_count * std::max(1u, config.global_fracture_seed_copies));
}

uint32_t nearest_global_fracture_chunk_id(Vec3 direction, const MarchingCubesConfig& config) {
    const uint32_t seed_count = global_fracture_seed_count(config);
    const Vec3 sample_direction = normalize(direction);
    uint32_t nearest_seed = 0;
    float best_score = -2.0f;
    for (uint32_t seed_id = 0; seed_id < seed_count; ++seed_id) {
        const float score = dot(sample_direction, global_fracture_seed_position(seed_id, seed_count, config));
        if (score > best_score) {
            best_score = score;
            nearest_seed = seed_id;
        }
    }
    return nearest_seed + 1u;
}

Vec3 plate_guided_cave_anchor_direction(
    uint32_t anchor_index,
    uint32_t anchor_count,
    const MarchingCubesConfig& config,
    const std::vector<Vec3>& plate_centers
) {
    const uint32_t seed = config.voxel_features.seed ^ 0x9d06e9c5u;
    constexpr float GoldenAngle = 2.39996323f;
    const float count = static_cast<float>(std::max(1u, anchor_count));
    const float row_jitter = (hash_unit(seed ^ (anchor_index * 0x7feb352du)) - 0.5f) * 0.72f;
    const float row = std::clamp(static_cast<float>(anchor_index) + 0.5f + row_jitter, 0.5f, count - 0.5f);
    const float z = 1.0f - 2.0f * (row / count);
    const float radius = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float phase = hash_unit(seed ^ 0x68bc21ebu) * (Pi * 2.0f);
    const float theta_jitter = (hash_unit(seed ^ (anchor_index * 0x846ca68bu) ^ 0xa511e9b3u) - 0.5f) * 0.35f;
    Vec3 direction = normalize({
        std::cos(static_cast<float>(anchor_index) * GoldenAngle + phase + theta_jitter) * radius,
        z,
        std::sin(static_cast<float>(anchor_index) * GoldenAngle + phase + theta_jitter) * radius,
    });

    const uint32_t plate_count = static_cast<uint32_t>(plate_centers.size());
    uint32_t nearest_plate = 0u;
    float best_score = -2.0f;
    float second_score = -2.0f;
    for (uint32_t plate_id = 0; plate_id < plate_count; ++plate_id) {
        const float score = dot(direction, plate_centers[plate_id]);
        if (score > best_score) {
            second_score = best_score;
            best_score = score;
            nearest_plate = plate_id;
        } else if (score > second_score) {
            second_score = score;
        }
    }

    const Vec3 plate_center = plate_centers[nearest_plate];
    const Frame plate_frame = build_frame(plate_center);
    const uint32_t plate_seed = seed ^
        (config.fracture_seed * 0x27d4eb2du) ^
        (nearest_plate * 0x85ebca6bu) ^
        (anchor_index * 0xc2b2ae35u);
    const float plate_margin = std::clamp((best_score - second_score) * static_cast<float>(plate_count) * 0.35f, 0.0f, 1.0f);
    const float cave_grain = 0.0016f + hash_unit(plate_seed ^ 0x51ed270bu) * 0.0014f;
    const float grain_angle = hash_unit(plate_seed ^ 0x165667b1u) * (Pi * 2.0f);
    const float grain_radius = cave_grain * (0.35f + plate_margin * 0.65f);
    const Vec3 tangent_grain =
        plate_frame.tangent * (std::cos(grain_angle) * grain_radius) +
        plate_frame.bitangent * (std::sin(grain_angle) * grain_radius);
    return normalize(direction + tangent_grain);
}

std::vector<Vec3> build_cave_anchor_points(const MarchingCubesConfig& config) {
    std::vector<Vec3> anchors;
    if (!config.voxel_features.enabled || config.voxel_features.cave_anchor_count == 0u) {
        return anchors;
    }

    anchors.reserve(config.voxel_features.cave_anchor_count);
    const uint32_t plate_count = global_fracture_seed_count(config);
    std::vector<Vec3> plate_centers;
    plate_centers.reserve(plate_count);
    for (uint32_t plate_id = 0; plate_id < plate_count; ++plate_id) {
        plate_centers.push_back(global_fracture_seed_position(plate_id, plate_count, config));
    }

    for (uint32_t anchor_index = 0; anchor_index < config.voxel_features.cave_anchor_count; ++anchor_index) {
        anchors.push_back(plate_guided_cave_anchor_direction(
            anchor_index,
            config.voxel_features.cave_anchor_count,
            config,
            plate_centers
        ));
    }
    return anchors;
}

void build_global_fracture_seeds(
    const GoldbergTopology& topology,
    Vec3 cell_center,
    const Frame& frame,
    const MarchingCubesConfig& config,
    std::vector<FractureSeed>& seeds
) {
    (void)topology;
    seeds.clear();
    const uint32_t seed_count = global_fracture_seed_count(config);
    seeds.reserve(seed_count);
    for (uint32_t seed_id = 0; seed_id < seed_count; ++seed_id) {
        const Vec3 seed_position = global_fracture_seed_position(seed_id, seed_count, config);
        const Vec2 local_seed = project_to_cell_plane(cell_center, frame, seed_position);
        seeds.push_back({
            local_seed,
            seed_position,
            seed_id + 1u,
            length2(local_seed),
        });
    }
}

void build_fracture_shards(
    const GoldbergTopology& topology,
    const GoldbergCell& cell,
    uint32_t cell_id,
    Vec3 cell_center,
    const Frame& frame,
    const std::vector<Vec2>& cell_polygon,
    const MarchingCubesConfig& config,
    std::vector<FractureSeed>& seeds,
    std::vector<FractureShard>& shards
) {
    const uint32_t shard_count = std::max(1u, cell.kind == GoldbergCellKind::Pentagon ? config.shards_per_pent : config.shards_per_hex);
    seeds.clear();
    seeds.reserve(shard_count);

    if (config.connect_fractures_across_cells) {
        build_global_fracture_seeds(topology, cell_center, frame, config, seeds);
    } else {
        seeds.push_back({polygon_centroid(cell_polygon), cell_center, cell_id * 32u + 1u, 0.0f});
        uint32_t attempt = 0;
        while (seeds.size() < shard_count && attempt < shard_count * 96u) {
            const Vec2 candidate = deterministic_point_in_polygon(
                cell_polygon,
                config.fracture_seed ^ (cell_id * 0x9e3779b9u) ^ (attempt * 0x85ebca6bu)
            );
            if (point_in_convex_polygon(candidate, cell_polygon)) {
                seeds.push_back({
                    candidate,
                    normalize(local_to_world(cell_center, frame, {candidate.x, candidate.y, 0.0f})),
                    cell_id * 32u + static_cast<uint32_t>(seeds.size()) + 1u,
                    length2(candidate),
                });
            }
            ++attempt;
        }

        while (seeds.size() < shard_count) {
            const uint32_t corner = static_cast<uint32_t>(seeds.size()) % static_cast<uint32_t>(cell_polygon.size());
            const Vec2 local_seed = lerp2(seeds.front().local, cell_polygon[corner], 0.55f);
            seeds.push_back({
                local_seed,
                normalize(local_to_world(cell_center, frame, {local_seed.x, local_seed.y, 0.0f})),
                cell_id * 32u + static_cast<uint32_t>(seeds.size()) + 1u,
                length2(local_seed),
            });
        }
    }

    shards.clear();
    shards.reserve(seeds.size());
    for (uint32_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
        std::vector<Vec2> shard_polygon = cell_polygon;
        for (uint32_t other_index = 0; other_index < seeds.size(); ++other_index) {
            if (other_index == seed_index) {
                continue;
            }
            if (config.connect_fractures_across_cells) {
                shard_polygon = clip_polygon_to_spherical_voronoi_half_plane(
                    shard_polygon,
                    cell_center,
                    frame,
                    seeds[seed_index].direction,
                    seeds[other_index].direction
                );
            } else {
                shard_polygon = clip_polygon_to_voronoi_half_plane(shard_polygon, seeds[seed_index].local, seeds[other_index].local);
            }
            if (shard_polygon.size() < 3) {
                break;
            }
        }
        if (shard_polygon.size() < 3 || std::fabs(signed_area(shard_polygon)) < 0.000001f) {
            continue;
        }

        shards.push_back({
            shard_polygon,
            seeds[seed_index].local,
            seeds[seed_index].chunk_id,
        });
    }
}

FracturedLocalSample fractured_local_sample(
    Vec2 local,
    Vec2 shard_center,
    uint32_t chunk_id,
    const std::vector<Vec2>& cell_polygon,
    float surface_radius,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    Vec2 adjusted = local;
    float radius = surface_radius;
    bool is_internal = false;
    const Vec2 to_center = shard_center - local;
    const float to_center_length = length2(to_center);
    if (to_center_length > 0.000001f) {
        adjusted = adjusted + to_center * (std::min(config.fracture_gap, to_center_length * 0.45f) / to_center_length);
        is_internal = true;
    }
    const float deterministic_lift = fracture_chunk_lift(chunk_id, config, cache);
    radius = std::max(0.1f, radius - config.fracture_depth + deterministic_lift);

    return {adjusted, radius, is_internal};
}

float fracture_chunk_top_radius(float surface_radius, uint32_t chunk_id, const MarchingCubesConfig& config, const FractureBuildCache& cache) {
    const float deterministic_lift = fracture_chunk_lift(chunk_id, config, cache);
    return std::max(0.1f, surface_radius - config.fracture_depth + deterministic_lift);
}

Vec3 fractured_sample_to_world(Vec2 adjusted, Vec3 cell_center, const Frame& frame, float radius) {
    return normalize(local_to_world(cell_center, frame, {adjusted.x, adjusted.y, 0.0f})) * radius;
}

Vec3 fractured_local_to_world(
    Vec2 local,
    Vec2 shard_center,
    uint32_t chunk_id,
    Vec3 cell_center,
    const Frame& frame,
    const std::vector<Vec2>& cell_polygon,
    float surface_radius,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    const FracturedLocalSample sample = fractured_local_sample(local, shard_center, chunk_id, cell_polygon, surface_radius, config, cache);
    return fractured_sample_to_world(sample.adjusted, cell_center, frame, sample.top_radius);
}

void emit_fracture_wall_quad(
    QuantizedMesh& mesh,
    uint32_t cell_id,
    Vec3 top_a,
    Vec3 top_b,
    Vec3 bottom_a,
    Vec3 bottom_b,
    uint32_t material_id,
    uint32_t fracture_chunk_id
) {
    Vec3 normal = cross(bottom_a - top_a, top_b - top_a);
    if (length(normal) <= 0.000001f) {
        normal = normalize((top_a + top_b) * 0.5f);
    } else {
        normal = normalize(normal);
    }

    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({top_a, normal, material_id, cell_id, fracture_chunk_id});
    mesh.vertices.push_back({bottom_a, normal, material_id, cell_id, fracture_chunk_id});
    mesh.vertices.push_back({top_b, normal, material_id, cell_id, fracture_chunk_id});
    mesh.vertices.push_back({bottom_b, normal, material_id, cell_id, fracture_chunk_id});

    mesh.triangle_indices.push_back(base);
    mesh.triangle_indices.push_back(base + 1);
    mesh.triangle_indices.push_back(base + 2);
    mesh.triangle_indices.push_back(base + 2);
    mesh.triangle_indices.push_back(base + 1);
    mesh.triangle_indices.push_back(base + 3);

    mesh.line_indices.push_back(base);
    mesh.line_indices.push_back(base + 2);
    mesh.line_indices.push_back(base + 2);
    mesh.line_indices.push_back(base + 3);
    mesh.line_indices.push_back(base + 3);
    mesh.line_indices.push_back(base + 1);
    mesh.line_indices.push_back(base + 1);
    mesh.line_indices.push_back(base);

    mesh.triangle_count += 2;
}

bool fracture_wall_point_is_internal(Vec2 point, const std::vector<Vec2>& cell_polygon, const MarchingCubesConfig& config) {
    (void)config;
    return point_in_convex_polygon(point, cell_polygon) && distance_to_polygon_boundary(point, cell_polygon) > 0.000001f;
}

bool edge_lies_on_cell_boundary(Vec2 a, Vec2 b, const std::vector<Vec2>& cell_polygon) {
    constexpr float BoundaryEpsilon = 0.0015f;
    const Vec2 midpoint = (a + b) * 0.5f;
    return distance_to_polygon_boundary(a, cell_polygon) <= BoundaryEpsilon &&
        distance_to_polygon_boundary(b, cell_polygon) <= BoundaryEpsilon &&
        distance_to_polygon_boundary(midpoint, cell_polygon) <= BoundaryEpsilon;
}

Vec2 fracture_wall_guard_crossing(
    Vec2 internal_point,
    Vec2 guarded_point,
    const std::vector<Vec2>& cell_polygon,
    const MarchingCubesConfig& config
) {
    float lo = 0.0f;
    float hi = 1.0f;
    for (uint32_t i = 0; i < 12; ++i) {
        const float mid = (lo + hi) * 0.5f;
        const Vec2 candidate = lerp2(internal_point, guarded_point, mid);
        if (fracture_wall_point_is_internal(candidate, cell_polygon, config)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lerp2(internal_point, guarded_point, lo);
}

void emit_fracture_wall_segment(
    QuantizedMesh& mesh,
    uint32_t cell_id,
    const FractureShard& shard,
    Vec2 a,
    Vec2 b,
    Vec3 cell_center,
    const Frame& frame,
    const std::vector<Vec2>& cell_polygon,
    float surface_radius,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    const FracturedLocalSample sample_a = fractured_local_sample(a, shard.center, shard.chunk_id, cell_polygon, surface_radius, config, cache);
    const FracturedLocalSample sample_b = fractured_local_sample(b, shard.center, shard.chunk_id, cell_polygon, surface_radius, config, cache);
    if (!sample_a.is_internal || !sample_b.is_internal) {
        return;
    }

    const float bottom_radius = std::max(0.1f, surface_radius - config.fracture_wall_depth);
    const Vec3 top_a = fractured_sample_to_world(sample_a.adjusted, cell_center, frame, sample_a.top_radius);
    const Vec3 top_b = fractured_sample_to_world(sample_b.adjusted, cell_center, frame, sample_b.top_radius);
    const Vec3 bottom_a = fractured_sample_to_world(sample_a.adjusted, cell_center, frame, bottom_radius);
    const Vec3 bottom_b = fractured_sample_to_world(sample_b.adjusted, cell_center, frame, bottom_radius);
    emit_fracture_wall_quad(
        mesh,
        cell_id,
        top_a,
        top_b,
        bottom_a,
        bottom_b,
        config.fracture_wall_material_id,
        shard.chunk_id
    );
}

void emit_trimmed_fracture_wall_edge(
    QuantizedMesh& mesh,
    uint32_t cell_id,
    const FractureShard& shard,
    Vec2 edge_a,
    Vec2 edge_b,
    Vec3 cell_center,
    const Frame& frame,
    const std::vector<Vec2>& cell_polygon,
    float surface_radius,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    constexpr uint32_t SegmentCount = 48;
    for (uint32_t step = 0; step < SegmentCount; ++step) {
        Vec2 segment_a = lerp2(edge_a, edge_b, static_cast<float>(step) / static_cast<float>(SegmentCount));
        Vec2 segment_b = lerp2(edge_a, edge_b, static_cast<float>(step + 1) / static_cast<float>(SegmentCount));
        const bool a_internal = fracture_wall_point_is_internal(segment_a, cell_polygon, config);
        const bool b_internal = fracture_wall_point_is_internal(segment_b, cell_polygon, config);
        if (!a_internal && !b_internal) {
            continue;
        }

        if (!a_internal) {
            segment_a = fracture_wall_guard_crossing(segment_b, segment_a, cell_polygon, config);
        }
        if (!b_internal) {
            segment_b = fracture_wall_guard_crossing(segment_a, segment_b, cell_polygon, config);
        }

        if (length2(segment_b - segment_a) > 0.000001f) {
            emit_fracture_wall_segment(
                mesh,
                cell_id,
                shard,
                segment_a,
                segment_b,
                cell_center,
                frame,
                cell_polygon,
                surface_radius,
                config,
                cache
            );
        }
    }
}

void emit_fracture_shard_walls(
    QuantizedMesh& mesh,
    uint32_t cell_id,
    const FractureShard& shard,
    Vec3 cell_center,
    const Frame& frame,
    const std::vector<Vec2>& cell_polygon,
    float surface_radius,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    if (!config.enable_fracture_walls || shard.polygon.size() < 3 || config.fracture_wall_depth <= 0.0f) {
        return;
    }

    for (uint32_t i = 0; i < shard.polygon.size(); ++i) {
        const Vec2 a = shard.polygon[i];
        const Vec2 b = shard.polygon[(i + 1) % shard.polygon.size()];
        if (edge_lies_on_cell_boundary(a, b, cell_polygon)) {
            continue;
        }
        emit_trimmed_fracture_wall_edge(
            mesh,
            cell_id,
            shard,
            a,
            b,
            cell_center,
            frame,
            cell_polygon,
            surface_radius,
            config,
            cache
        );
    }
}

bool point_inside_cave_entrance(Vec3 position, const LocalVoxelFeature& feature, float scale = 1.0f) {
    const Vec3 offset = position - feature.center_mesh;
    const float tangent_distance = std::sqrt(
        dot(offset, feature.tangent_mesh) * dot(offset, feature.tangent_mesh) +
        dot(offset, feature.bitangent_mesh) * dot(offset, feature.bitangent_mesh)
    );
    return tangent_distance <= kilometers_to_world_units(feature.entrance_radius_km) * scale;
}

bool local_triangle_inside_cave_entrance(
    const std::array<Vec2, 3>& triangle,
    Vec3 cell_center,
    const Frame& frame,
    const LocalVoxelFeature& feature
) {
    const Vec2 centroid = (triangle[0] + triangle[1] + triangle[2]) * (1.0f / 3.0f);
    const Vec3 position = normalize(local_to_world(cell_center, frame, {centroid.x, centroid.y, 0.0f}));
    return point_inside_cave_entrance(position, feature, 0.98f);
}

bool local_triangle_inside_any_cave_entrance(
    const std::array<Vec2, 3>& triangle,
    Vec3 cell_center,
    const Frame& frame,
    const std::vector<const LocalVoxelFeature*>& cave_features
) {
    for (const LocalVoxelFeature* feature : cave_features) {
        if (feature != nullptr && local_triangle_inside_cave_entrance(triangle, cell_center, frame, *feature)) {
            return true;
        }
    }
    return false;
}

void append_cave_rim_quad(
    QuantizedMesh& mesh,
    uint32_t cell_id,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    Vec3 d
) {
    Vec3 normal = cross(b - a, c - a);
    if (length(normal) <= 0.000001f) {
        normal = normalize((a + b + c + d) * 0.25f);
    } else {
        normal = normalize(normal);
    }

    constexpr uint32_t CaveRimMaterialId = 6u;
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({a, normal, CaveRimMaterialId, cell_id, 0u});
    mesh.vertices.push_back({b, normal, CaveRimMaterialId, cell_id, 0u});
    mesh.vertices.push_back({c, normal, CaveRimMaterialId, cell_id, 0u});
    mesh.vertices.push_back({d, normal, CaveRimMaterialId, cell_id, 0u});

    mesh.triangle_indices.push_back(base);
    mesh.triangle_indices.push_back(base + 1u);
    mesh.triangle_indices.push_back(base + 2u);
    mesh.triangle_indices.push_back(base);
    mesh.triangle_indices.push_back(base + 2u);
    mesh.triangle_indices.push_back(base + 3u);

    mesh.line_indices.push_back(base);
    mesh.line_indices.push_back(base + 1u);
    mesh.line_indices.push_back(base + 1u);
    mesh.line_indices.push_back(base + 2u);
    mesh.line_indices.push_back(base + 2u);
    mesh.line_indices.push_back(base + 3u);
    mesh.line_indices.push_back(base + 3u);
    mesh.line_indices.push_back(base);
    mesh.triangle_count += 2u;
}

void emit_cave_entrance_rim(
    QuantizedMesh& mesh,
    uint32_t cell_id,
    const LocalVoxelFeature& feature,
    float surface_radius
) {
    constexpr uint32_t SegmentCount = 40u;
    const float inner_radius = kilometers_to_world_units(feature.entrance_radius_km) * 0.98f;
    const float outer_radius = inner_radius * 1.26f;
    const float inward_depth = std::max(kilometers_to_world_units(5.0f), inner_radius * 0.25f);
    for (uint32_t i = 0; i < SegmentCount; ++i) {
        const float a0 = (2.0f * Pi * static_cast<float>(i)) / static_cast<float>(SegmentCount);
        const float a1 = (2.0f * Pi * static_cast<float>(i + 1u)) / static_cast<float>(SegmentCount);
        auto ring_point = [&](float radius, float angle, float radial_offset) {
            const Vec3 tangent_offset =
                feature.tangent_mesh * (std::cos(angle) * radius) +
                feature.bitangent_mesh * (std::sin(angle) * radius);
            return normalize(feature.center_mesh + tangent_offset) * (surface_radius + radial_offset);
        };

        const Vec3 outer0 = ring_point(outer_radius, a0, 0.0f);
        const Vec3 outer1 = ring_point(outer_radius, a1, 0.0f);
        const Vec3 inner0 = ring_point(inner_radius, a0, -inward_depth);
        const Vec3 inner1 = ring_point(inner_radius, a1, -inward_depth);
        append_cave_rim_quad(mesh, cell_id, outer0, outer1, inner1, inner0);
    }
}

void emit_fractured_local_triangle(
    QuantizedMesh& mesh,
    BoundaryEdgeMap& boundary_edges,
    uint32_t cell_id,
    const std::array<Vec2, 3>& triangle,
    const FractureShard& shard,
    Vec3 cell_center,
    Vec3 cell_normal,
    const Frame& frame,
    const std::vector<Vec2>& cell_polygon,
    float surface_radius,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache,
    uint32_t material_id
) {
    const std::array<GridSample, 3> clip_triangle = {{
        {{triangle[0].x, triangle[0].y, 0.0f}, {}, 0.0f},
        {{triangle[1].x, triangle[1].y, 0.0f}, {}, 0.0f},
        {{triangle[2].x, triangle[2].y, 0.0f}, {}, 0.0f},
    }};
    const std::vector<ClipVertex> clipped = clip_to_cell_polygon(clip_triangle, shard.polygon);
    if (clipped.size() < 3) {
        return;
    }

    const Vec2 origin_local = {clipped.front().local.x, clipped.front().local.y};
    const Vec3 fan_origin = fractured_local_to_world(origin_local, shard.center, shard.chunk_id, cell_center, frame, cell_polygon, surface_radius, config, cache);
    for (uint32_t i = 1; i + 1 < clipped.size(); ++i) {
        const Vec2 b_local = {clipped[i].local.x, clipped[i].local.y};
        const Vec2 c_local = {clipped[i + 1].local.x, clipped[i + 1].local.y};
        const Vec3 b = fractured_local_to_world(b_local, shard.center, shard.chunk_id, cell_center, frame, cell_polygon, surface_radius, config, cache);
        const Vec3 c = fractured_local_to_world(c_local, shard.center, shard.chunk_id, cell_center, frame, cell_polygon, surface_radius, config, cache);

        std::array<Vec2, 3> local_points = {{origin_local, b_local, c_local}};
        std::array<Vec3, 3> world_points = {{fan_origin, b, c}};
        if (dot(cross(world_points[1] - world_points[0], world_points[2] - world_points[0]), cell_normal) < 0.0f) {
            std::swap(local_points[1], local_points[2]);
            std::swap(world_points[1], world_points[2]);
        }

        const Vec3 normal = normalize(cross(world_points[1] - world_points[0], world_points[2] - world_points[0]));
        const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({world_points[0], normal, material_id, cell_id, shard.chunk_id});
        mesh.vertices.push_back({world_points[1], normal, material_id, cell_id, shard.chunk_id});
        mesh.vertices.push_back({world_points[2], normal, material_id, cell_id, shard.chunk_id});
        mesh.triangle_indices.push_back(base);
        mesh.triangle_indices.push_back(base + 1);
        mesh.triangle_indices.push_back(base + 2);
        mesh.line_indices.push_back(base);
        mesh.line_indices.push_back(base + 1);
        mesh.line_indices.push_back(base + 1);
        mesh.line_indices.push_back(base + 2);
        mesh.line_indices.push_back(base + 2);
        mesh.line_indices.push_back(base);
        ++mesh.triangle_count;

        constexpr float BoundaryRecordEpsilon = 0.0015f;
        for (uint32_t edge = 0; edge < 3; ++edge) {
            const uint32_t next = (edge + 1) % 3;
            if (distance_to_polygon_boundary(local_points[edge], cell_polygon) <= BoundaryRecordEpsilon &&
                distance_to_polygon_boundary(local_points[next], cell_polygon) <= BoundaryRecordEpsilon) {
                record_triangle_edge(boundary_edges, cell_id, world_points[edge], world_points[next], material_id, shard.chunk_id);
            }
        }
    }
}

void emit_subdivided_goldberg_cell_plane(
    QuantizedMesh& mesh,
    BoundaryEdgeMap& boundary_edges,
    const GoldbergTopology& topology,
    uint32_t cell_id,
    uint32_t subdivisions,
    float surface_radius,
    uint32_t material_id,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache,
    std::vector<FractureSeed>& fracture_seed_scratch,
    std::vector<FractureShard>& fracture_shard_scratch,
    const std::vector<LocalVoxelFeature>& voxel_features
) {
    const GoldbergCell& cell = topology.cells[cell_id];
    if (cell.corner_indices.size() < 3 || subdivisions == 0) {
        return;
    }

    const Frame frame = build_frame(cell.normal);
    const std::vector<Vec2> cell_polygon = cell_clip_polygon(topology, cell, cell.center, frame);
    std::vector<FractureShard>& shards = fracture_shard_scratch;
    shards.clear();
    if (config.enable_fractures) {
        build_fracture_shards(topology, cell, cell_id, cell.center, frame, cell_polygon, config, fracture_seed_scratch, shards);
    }
    if (config.enable_fractures && config.enable_fracture_walls) {
        for (const FractureShard& shard : shards) {
            emit_fracture_shard_walls(mesh, cell_id, shard, cell.center, frame, cell_polygon, surface_radius, config, cache);
        }
    }
    std::vector<const LocalVoxelFeature*> cell_cave_features;
    for (const LocalVoxelFeature& feature : voxel_features) {
        if (feature.kind == VoxelFeatureKind::CaveSystem && feature.owner_cell_id == cell_id) {
            cell_cave_features.push_back(&feature);
        }
    }
    for (const LocalVoxelFeature* cave_feature : cell_cave_features) {
        emit_cave_entrance_rim(mesh, cell_id, *cave_feature, surface_radius);
    }

    for (uint32_t corner_slot = 0; corner_slot < cell.corner_indices.size(); ++corner_slot) {
        const uint32_t next_slot = (corner_slot + 1) % static_cast<uint32_t>(cell.corner_indices.size());
        const Vec3 center = cell.center;
        const Vec3 edge_a = topology.vertices[cell.corner_indices[corner_slot]].position;
        const Vec3 edge_b = topology.vertices[cell.corner_indices[next_slot]].position;
        const Vec2 local_edge_a = project_to_cell_plane(cell.center, frame, edge_a);
        const Vec2 local_edge_b = project_to_cell_plane(cell.center, frame, edge_b);

        for (uint32_t a_step = 0; a_step < subdivisions; ++a_step) {
            for (uint32_t b_step = 0; b_step + a_step < subdivisions; ++b_step) {
                if (config.enable_fractures && !shards.empty()) {
                    const std::array<Vec2, 3> tri0 = {{
                        subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step, b_step, subdivisions),
                        subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step + 1, b_step, subdivisions),
                        subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step, b_step + 1, subdivisions),
                    }};
                    if (!local_triangle_inside_any_cave_entrance(tri0, cell.center, frame, cell_cave_features)) {
                        for (const FractureShard& shard : shards) {
                            emit_fractured_local_triangle(mesh, boundary_edges, cell_id, tri0, shard, cell.center, cell.normal, frame, cell_polygon, surface_radius, config, cache, material_id);
                        }
                    }
                } else {
                    const std::array<Vec2, 3> tri0 = {{
                        subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step, b_step, subdivisions),
                        subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step + 1, b_step, subdivisions),
                        subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step, b_step + 1, subdivisions),
                    }};
                    if (local_triangle_inside_any_cave_entrance(tri0, cell.center, frame, cell_cave_features)) {
                        continue;
                    }
                    const Vec3 p0 = subdivided_cell_vertex(center, edge_a, edge_b, a_step, b_step, subdivisions, surface_radius);
                    const Vec3 p1 = subdivided_cell_vertex(center, edge_a, edge_b, a_step + 1, b_step, subdivisions, surface_radius);
                    const Vec3 p2 = subdivided_cell_vertex(center, edge_a, edge_b, a_step, b_step + 1, subdivisions, surface_radius);
                    emit_oriented_mesh_triangle(mesh, boundary_edges, cell_id, p0, p1, p2, cell.normal, material_id);
                }

                if (a_step + b_step + 1 < subdivisions) {
                    if (config.enable_fractures && !shards.empty()) {
                        const std::array<Vec2, 3> tri1 = {{
                            subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step + 1, b_step, subdivisions),
                            subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step + 1, b_step + 1, subdivisions),
                            subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step, b_step + 1, subdivisions),
                        }};
                        if (!local_triangle_inside_any_cave_entrance(tri1, cell.center, frame, cell_cave_features)) {
                            for (const FractureShard& shard : shards) {
                                emit_fractured_local_triangle(mesh, boundary_edges, cell_id, tri1, shard, cell.center, cell.normal, frame, cell_polygon, surface_radius, config, cache, material_id);
                            }
                        }
                    } else {
                        const std::array<Vec2, 3> tri1 = {{
                            subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step + 1, b_step, subdivisions),
                            subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step + 1, b_step + 1, subdivisions),
                            subdivided_cell_local_vertex(local_edge_a, local_edge_b, a_step, b_step + 1, subdivisions),
                        }};
                        if (local_triangle_inside_any_cave_entrance(tri1, cell.center, frame, cell_cave_features)) {
                            continue;
                        }
                        const Vec3 p1 = subdivided_cell_vertex(center, edge_a, edge_b, a_step + 1, b_step, subdivisions, surface_radius);
                        const Vec3 p2 = subdivided_cell_vertex(center, edge_a, edge_b, a_step, b_step + 1, subdivisions, surface_radius);
                        const Vec3 p3 = subdivided_cell_vertex(center, edge_a, edge_b, a_step + 1, b_step + 1, subdivisions, surface_radius);
                        emit_oriented_mesh_triangle(mesh, boundary_edges, cell_id, p1, p3, p2, cell.normal, material_id);
                    }
                }
            }
        }
    }
}

uint32_t lerp_uint32(uint32_t a, uint32_t b, float t) {
    return static_cast<uint32_t>(std::round(static_cast<float>(a) * (1.0f - t) + static_cast<float>(b) * t));
}

uint32_t cell_lod_subdivisions(const GoldbergCell& cell, const MarchingCubesConfig& config, uint32_t fallback_subdivisions) {
    if (!config.enable_lod_subdivision_test && !config.enable_camera_proximity_lod) {
        return fallback_subdivisions;
    }

    const uint32_t min_subdivisions = std::max(1u, std::min(config.lod_min_subdivisions, config.lod_max_subdivisions));
    const uint32_t max_subdivisions = std::max(min_subdivisions, std::max(config.lod_min_subdivisions, config.lod_max_subdivisions));
    const uint32_t level_count = std::max(1u, config.lod_levels);
    if (level_count == 1 || min_subdivisions == max_subdivisions) {
        return min_subdivisions;
    }

    float lod_score = 0.0f;
    if (config.enable_camera_proximity_lod) {
        const Vec3 lod_focus = normalize(config.lod_camera_position);
        const Vec3 cell_surface_position = normalize(cell.center);
        const float surface_distance = length(cell_surface_position - lod_focus);
        const float inner_radius = std::max(0.0f, std::min(config.lod_inner_patch_radius, config.lod_outer_patch_radius));
        const float outer_radius = std::max(inner_radius + 0.0001f, std::max(config.lod_inner_patch_radius, config.lod_outer_patch_radius));
        lod_score = std::clamp((outer_radius - surface_distance) / (outer_radius - inner_radius), 0.0f, 1.0f);
    } else {
        const Vec3 lod_focus = normalize(Vec3{0.42f, 0.18f, 1.0f});
        lod_score = std::clamp((dot(cell.center, lod_focus) + 1.0f) * 0.5f, 0.0f, 1.0f);
    }

    const uint32_t level = std::min(level_count - 1, static_cast<uint32_t>(std::floor(lod_score * static_cast<float>(level_count))));
    const float t = static_cast<float>(level) / static_cast<float>(level_count - 1);
    return std::max(1u, lerp_uint32(min_subdivisions, max_subdivisions, t));
}

void append_stitch_triangle(
    QuantizedMesh& mesh,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    uint32_t cell_id,
    uint32_t material_id = 2u,
    uint32_t fracture_chunk_id = 0
) {
    const Vec3 normal = normalize(cross(b - a, c - a));
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());

    mesh.vertices.push_back({a, normal, material_id, cell_id, fracture_chunk_id});
    mesh.vertices.push_back({b, normal, material_id, cell_id, fracture_chunk_id});
    mesh.vertices.push_back({c, normal, material_id, cell_id, fracture_chunk_id});

    mesh.stitch_triangle_indices.push_back(base);
    mesh.stitch_triangle_indices.push_back(base + 1);
    mesh.stitch_triangle_indices.push_back(base + 2);
    mesh.stitch_line_indices.push_back(base);
    mesh.stitch_line_indices.push_back(base + 1);
    mesh.stitch_line_indices.push_back(base + 1);
    mesh.stitch_line_indices.push_back(base + 2);
    mesh.stitch_line_indices.push_back(base + 2);
    mesh.stitch_line_indices.push_back(base);
    ++mesh.stitch_triangle_count;
}

void append_chain_stitch_triangle(
    QuantizedMesh& mesh,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    uint32_t cell_id,
    uint32_t material_id = 2u,
    uint32_t fracture_chunk_id = 0
) {
    append_stitch_triangle(mesh, a, b, c, cell_id, material_id, fracture_chunk_id);
    ++mesh.chain_stitch_triangle_count;
}

void append_fallback_stitch_triangle(QuantizedMesh& mesh, Vec3 a, Vec3 b, Vec3 c, uint32_t cell_id) {
    append_stitch_triangle(mesh, a, b, c, cell_id);
    ++mesh.fallback_stitch_triangle_count;
}

std::array<uint32_t, 2> shared_goldberg_edge(const GoldbergCell& a, const GoldbergCell& b) {
    std::array<uint32_t, 2> shared = {UINT32_MAX, UINT32_MAX};
    uint32_t count = 0;

    for (uint32_t a_corner : a.corner_indices) {
        if (std::find(b.corner_indices.begin(), b.corner_indices.end(), a_corner) == b.corner_indices.end()) {
            continue;
        }
        if (count < shared.size()) {
            shared[count] = a_corner;
        }
        ++count;
    }

    if (count != 2) {
        return {UINT32_MAX, UINT32_MAX};
    }
    return shared;
}

float edge_sort_value(Vec3 point, Vec3 edge_start, Vec3 edge_axis) {
    return dot(point - edge_start, edge_axis);
}

std::vector<Vec3> resample_chain(const std::vector<Vec3>& chain, uint32_t desired_count, float radius) {
    if (chain.empty() || desired_count == 0) {
        return {};
    }
    if (chain.size() == 1 || desired_count == 1) {
        return {normalize(chain.front()) * radius};
    }

    std::vector<float> distances(chain.size(), 0.0f);
    for (uint32_t i = 1; i < chain.size(); ++i) {
        distances[i] = distances[i - 1] + length(chain[i] - chain[i - 1]);
    }

    const float total_distance = distances.back();
    if (total_distance <= 0.000001f) {
        return {normalize(chain.front()) * radius, normalize(chain.back()) * radius};
    }

    std::vector<Vec3> resampled;
    resampled.reserve(desired_count);
    uint32_t segment = 1;
    for (uint32_t i = 0; i < desired_count; ++i) {
        const float target = total_distance * static_cast<float>(i) / static_cast<float>(desired_count - 1);
        while (segment + 1 < distances.size() && distances[segment] < target) {
            ++segment;
        }

        const float start_distance = distances[segment - 1];
        const float end_distance = distances[segment];
        const float span = end_distance - start_distance;
        const float t = span > 0.000001f ? (target - start_distance) / span : 0.0f;
        resampled.push_back(normalize(lerp(chain[segment - 1], chain[segment], t)) * radius);
    }

    return resampled;
}

bool shared_boundary_geometry(
    const GoldbergTopology& topology,
    uint32_t cell_id,
    uint32_t neighbor_id,
    BoundaryPairGeometry& geometry
) {
    const std::array<uint32_t, 2> shared_edge = shared_goldberg_edge(topology.cells[cell_id], topology.cells[neighbor_id]);
    if (shared_edge[0] == UINT32_MAX || shared_edge[1] == UINT32_MAX) {
        return false;
    }

    geometry.edge_a = topology.vertices[shared_edge[0]].position;
    geometry.edge_b = topology.vertices[shared_edge[1]].position;
    const Vec3 center_mid = normalize(topology.cells[cell_id].center + topology.cells[neighbor_id].center);
    if (dot(normalize(geometry.edge_a + geometry.edge_b), center_mid) < 0.0f) {
        std::swap(geometry.edge_a, geometry.edge_b);
    }
    geometry.edge_axis = normalize(geometry.edge_b - geometry.edge_a);
    geometry.edge_length = length(geometry.edge_b - geometry.edge_a);
    return geometry.edge_length > 0.000001f;
}

bool boundary_segment_shares_endpoint(const BoundarySegment& segment, const std::vector<BoundarySegment>& component) {
    const PositionKey a0 = position_key(segment.a);
    const PositionKey b0 = position_key(segment.b);

    for (const BoundarySegment& other : component) {
        const PositionKey a1 = position_key(other.a);
        const PositionKey b1 = position_key(other.b);
        if (a0 == a1 || a0 == b1 || b0 == a1 || b0 == b1) {
            return true;
        }
    }

    return false;
}

BoundaryRun make_boundary_run(
    const std::vector<BoundarySegment>& component,
    const BoundaryPairGeometry& geometry,
    float radius
) {
    std::map<PositionKey, Vec3> node_positions;
    std::map<PositionKey, std::vector<PositionKey>> adjacency;
    for (const BoundarySegment& segment : component) {
        const Vec3 a = normalize(segment.a) * radius;
        const Vec3 b = normalize(segment.b) * radius;
        const PositionKey key_a = position_key(a);
        const PositionKey key_b = position_key(b);
        node_positions[key_a] = a;
        node_positions[key_b] = b;
        adjacency[key_a].push_back(key_b);
        adjacency[key_b].push_back(key_a);
    }

    if (node_positions.size() < 2) {
        return {};
    }

    PositionKey start_key = node_positions.begin()->first;
    float best_start_sort = edge_sort_value(node_positions.begin()->second, geometry.edge_a, geometry.edge_axis);
    for (const auto& entry : node_positions) {
        const uint32_t degree = static_cast<uint32_t>(adjacency[entry.first].size());
        const float sort_value = edge_sort_value(entry.second, geometry.edge_a, geometry.edge_axis);
        const bool current_is_open_end = adjacency[start_key].size() == 1;
        const bool candidate_is_open_end = degree == 1;
        if ((candidate_is_open_end && !current_is_open_end) || (candidate_is_open_end == current_is_open_end && sort_value < best_start_sort)) {
            start_key = entry.first;
            best_start_sort = sort_value;
        }
    }

    BoundaryRun run;
    run.points.reserve(node_positions.size());
    std::map<EdgeKey, bool> visited_edges;
    PositionKey current = start_key;
    PositionKey previous = start_key;
    bool has_previous = false;

    while (true) {
        run.points.push_back(node_positions[current]);
        const float current_sort = edge_sort_value(node_positions[current], geometry.edge_a, geometry.edge_axis);

        bool found_next = false;
        PositionKey next = current;
        float best_next_sort = 1000000.0f;
        for (const PositionKey& candidate : adjacency[current]) {
            const EdgeKey candidate_edge = {std::min(current, candidate), std::max(current, candidate)};
            if (visited_edges.find(candidate_edge) != visited_edges.end()) {
                continue;
            }
            if (has_previous && candidate == previous && adjacency[current].size() > 1) {
                continue;
            }

            const float candidate_sort = edge_sort_value(node_positions[candidate], geometry.edge_a, geometry.edge_axis);
            const float sort_penalty = candidate_sort < current_sort ? 1000.0f + (current_sort - candidate_sort) : candidate_sort - current_sort;
            if (!found_next || sort_penalty < best_next_sort) {
                found_next = true;
                next = candidate;
                best_next_sort = sort_penalty;
            }
        }

        if (!found_next) {
            break;
        }

        const EdgeKey traversed_edge = {std::min(current, next), std::max(current, next)};
        visited_edges[traversed_edge] = true;
        previous = current;
        current = next;
        has_previous = true;
    }

    if (run.points.size() < 2) {
        return {};
    }

    Vec3 midpoint_sum = {};
    run.min_sort = 1000000.0f;
    run.max_sort = -1000000.0f;
    for (const Vec3 point : run.points) {
        const float sort_value = edge_sort_value(point, geometry.edge_a, geometry.edge_axis);
        run.min_sort = std::min(run.min_sort, sort_value);
        run.max_sort = std::max(run.max_sort, sort_value);
        midpoint_sum = midpoint_sum + point;
    }

    run.midpoint = normalize(midpoint_sum / static_cast<float>(run.points.size())) * radius;

    return run;
}

std::vector<BoundaryRun> build_boundary_runs(
    QuantizedMesh& mesh,
    const std::vector<BoundarySegment>& segments,
    const BoundaryPairGeometry& geometry,
    float radius
) {
    std::vector<BoundaryRun> runs;
    std::vector<bool> used(segments.size(), false);

    for (uint32_t start = 0; start < segments.size(); ++start) {
        if (used[start]) {
            continue;
        }

        std::vector<BoundarySegment> component;
        component.push_back(segments[start]);
        used[start] = true;

        bool grew = true;
        while (grew) {
            grew = false;
            for (uint32_t i = 0; i < segments.size(); ++i) {
                if (used[i] || !boundary_segment_shares_endpoint(segments[i], component)) {
                    continue;
                }
                component.push_back(segments[i]);
                used[i] = true;
                grew = true;
            }
        }

        BoundaryRun run = make_boundary_run(component, geometry, radius);
        if (run.points.size() >= 2) {
            runs.push_back(run);
            ++mesh.boundary_run_count;
        }
    }

    return runs;
}

bool segment_is_near_shared_boundary(
    const GoldbergTopology& topology,
    uint32_t cell_id,
    uint32_t neighbor_id,
    Vec3 midpoint,
    const BoundaryPairGeometry& geometry
) {
    const float along_edge = edge_sort_value(midpoint, geometry.edge_a, geometry.edge_axis);
    const float along_margin = geometry.edge_length * 0.22f;
    if (along_edge < -along_margin || along_edge > geometry.edge_length + along_margin) {
        return false;
    }

    const float t = std::clamp(along_edge / geometry.edge_length, 0.0f, 1.0f);
    const Vec3 closest_edge_point = normalize(lerp(geometry.edge_a, geometry.edge_b, t));
    const float edge_distance = length(midpoint - closest_edge_point);
    const float max_edge_distance = std::max(geometry.edge_length * 0.42f, 0.055f);
    if (edge_distance > max_edge_distance) {
        return false;
    }

    const float current_dot = dot(midpoint, topology.cells[cell_id].center);
    const float neighbor_dot = dot(midpoint, topology.cells[neighbor_id].center);
    const float max_bisector_delta = 0.055f;
    return std::fabs(current_dot - neighbor_dot) <= max_bisector_delta;
}

bool boundary_edge_record_less(const BoundaryEdgeRecord& lhs, const BoundaryEdgeRecord& rhs) {
    if (lhs.cell_id != rhs.cell_id) {
        return lhs.cell_id < rhs.cell_id;
    }
    return lhs.edge < rhs.edge;
}

bool boundary_edge_record_same_key(const BoundaryEdgeRecord& lhs, const BoundaryEdgeRecord& rhs) {
    return lhs.cell_id == rhs.cell_id &&
        !(lhs.edge < rhs.edge) &&
        !(rhs.edge < lhs.edge);
}

void sort_and_merge_boundary_edges(BoundaryEdgeMap& boundary_edges) {
    if (boundary_edges.empty()) {
        return;
    }

    std::sort(boundary_edges.begin(), boundary_edges.end(), boundary_edge_record_less);
    uint32_t write_index = 0;
    for (uint32_t read_index = 1; read_index < boundary_edges.size(); ++read_index) {
        if (boundary_edge_record_same_key(boundary_edges[write_index], boundary_edges[read_index])) {
            boundary_edges[write_index].count += boundary_edges[read_index].count;
            continue;
        }
        ++write_index;
        if (write_index != read_index) {
            boundary_edges[write_index] = boundary_edges[read_index];
        }
    }
    boundary_edges.resize(write_index + 1);
}

BoundaryPairMap build_boundary_pair_chains(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const BoundaryEdgeMap& boundary_edges,
    const MarchingCubesConfig& config
) {
    BoundaryPairMap pairs;

    for (const BoundaryEdgeRecord& record : boundary_edges) {
        if (record.count != 1) {
            continue;
        }

        const Vec3 midpoint = normalize((record.a + record.b) * 0.5f);
        uint32_t accepted_neighbor = UINT32_MAX;
        float best_score = 1000000.0f;

        for (uint32_t neighbor_id : topology.cells[record.cell_id].neighbor_indices) {
            BoundaryPairGeometry geometry;
            if (!shared_boundary_geometry(topology, record.cell_id, neighbor_id, geometry)) {
                continue;
            }
            if (!segment_is_near_shared_boundary(topology, record.cell_id, neighbor_id, midpoint, geometry)) {
                continue;
            }

            const float along_edge = edge_sort_value(midpoint, geometry.edge_a, geometry.edge_axis);
            const float t = std::clamp(along_edge / geometry.edge_length, 0.0f, 1.0f);
            const Vec3 closest_edge_point = normalize(lerp(geometry.edge_a, geometry.edge_b, t));
            const float edge_distance = length(midpoint - closest_edge_point);
            const float bisector_delta = std::fabs(dot(midpoint, topology.cells[record.cell_id].center) - dot(midpoint, topology.cells[neighbor_id].center));
            const float score = edge_distance + bisector_delta * 2.0f;
            if (score < best_score) {
                best_score = score;
                accepted_neighbor = neighbor_id;
            }
        }

        if (accepted_neighbor == UINT32_MAX) {
            continue;
        }

        const BoundaryPairKey pair_key = boundary_pair_key(record.cell_id, accepted_neighbor);
        uint32_t fracture_chunk_id = record.fracture_chunk_id;
        if (config.enable_fractures && config.connect_fractures_across_cells && fracture_chunk_id != 0u) {
            fracture_chunk_id = nearest_global_fracture_chunk_id(midpoint, config);
        }
        BoundarySegment segment = {record.a, record.b, midpoint, 0.0f, record.material_id, fracture_chunk_id};
        BoundaryPairChains& chains = pairs[pair_key];
        if (record.cell_id == pair_key.a) {
            chains.a_segments.push_back(segment);
        } else {
            chains.b_segments.push_back(segment);
        }
        ++mesh.boundary_edge_count;
    }

    return pairs;
}

float run_overlap(const BoundaryRun& a, const BoundaryRun& b) {
    return std::min(a.max_sort, b.max_sort) - std::max(a.min_sort, b.min_sort);
}

bool runs_are_compatible(
    const BoundaryRun& a,
    const BoundaryRun& b,
    const BoundaryPairGeometry& geometry,
    float& score
) {
    const float overlap = run_overlap(a, b);
    const float min_overlap = std::max(geometry.edge_length * 0.035f, 0.008f);
    if (overlap < min_overlap) {
        return false;
    }

    const float midpoint_distance = length(a.midpoint - b.midpoint);
    const float max_midpoint_distance = std::max(geometry.edge_length * 0.58f, 0.070f);
    if (midpoint_distance > max_midpoint_distance) {
        return false;
    }

    score = midpoint_distance - overlap * 0.35f;
    return true;
}

std::vector<CorridorPoint> collect_corridor_points(
    const std::vector<BoundarySegment>& segments,
    const BoundaryPairGeometry& geometry,
    float radius
) {
    std::map<CorridorPointKey, CorridorPoint> unique_points;
    for (const BoundarySegment& segment : segments) {
        const std::array<Vec3, 2> endpoints = {{
            normalize(segment.a) * radius,
            normalize(segment.b) * radius,
        }};

        for (Vec3 endpoint : endpoints) {
            const float sort_value = edge_sort_value(endpoint, geometry.edge_a, geometry.edge_axis);
            const float t = std::clamp(sort_value / geometry.edge_length, 0.0f, 1.0f);
            const Vec3 edge_point = normalize(lerp(geometry.edge_a, geometry.edge_b, t)) * radius;
            const float max_corridor_distance = std::max(geometry.edge_length * 0.36f, 0.045f);
            if (sort_value < -geometry.edge_length * 0.12f ||
                sort_value > geometry.edge_length * 1.12f ||
                length(endpoint - edge_point) > max_corridor_distance) {
                continue;
            }

            unique_points[{position_key(endpoint), segment.fracture_chunk_id}] = {
                endpoint,
                sort_value,
                segment.material_id,
                segment.fracture_chunk_id,
            };
        }
    }

    std::vector<CorridorPoint> points;
    points.reserve(unique_points.size());
    for (const auto& entry : unique_points) {
        points.push_back(entry.second);
    }

    std::sort(points.begin(), points.end(), [](const CorridorPoint& a, const CorridorPoint& b) {
        return a.sort_value < b.sort_value;
    });

    return points;
}

bool append_greedy_corridor_stitches(
    QuantizedMesh& mesh,
    const std::vector<CorridorPoint>& side_a,
    const std::vector<CorridorPoint>& side_b,
    const BoundaryPairGeometry& geometry,
    uint32_t cell_id,
    float radius
) {
    struct Pair {
        Vec3 a;
        Vec3 b;
        float sort_value = 0.0f;
        uint32_t material_id = 0;
        uint32_t fracture_chunk_id = 0;
    };

    std::vector<Pair> pairs;
    pairs.reserve(std::min(side_a.size(), side_b.size()));
    const float max_sort_delta = std::max(geometry.edge_length * 0.18f, 0.020f);
    const float max_cross_gap = std::max(geometry.edge_length * 0.38f, 0.050f);
    uint32_t b_start = 0;

    for (const CorridorPoint& a : side_a) {
        while (b_start < side_b.size() && side_b[b_start].sort_value < a.sort_value - max_sort_delta) {
            ++b_start;
        }

        uint32_t best_b = UINT32_MAX;
        float best_score = 1000000.0f;
        for (uint32_t b_index = b_start; b_index < side_b.size(); ++b_index) {
            const CorridorPoint& b = side_b[b_index];
            if (a.fracture_chunk_id != b.fracture_chunk_id) {
                continue;
            }
            const float sort_delta = std::fabs(a.sort_value - b.sort_value);
            if (sort_delta > max_sort_delta) {
                if (b.sort_value > a.sort_value) {
                    break;
                }
                continue;
            }

            const float cross_gap = length(a.position - b.position);
            if (cross_gap > max_cross_gap) {
                ++mesh.rejected_greedy_jump_count;
                continue;
            }

            const float score = sort_delta + cross_gap * 0.45f;
            if (score < best_score) {
                best_score = score;
                best_b = b_index;
            }
        }

        if (best_b == UINT32_MAX) {
            continue;
        }

        pairs.push_back({
            normalize(a.position) * radius,
            normalize(side_b[best_b].position) * radius,
            (a.sort_value + side_b[best_b].sort_value) * 0.5f,
            a.material_id,
            a.fracture_chunk_id,
        });
        b_start = best_b + 1;
    }

    if (pairs.size() < 2) {
        return false;
    }

    const float max_path_step = std::max(geometry.edge_length * 0.30f, 0.040f);
    const float max_triangle_edge = std::max(geometry.edge_length * 0.48f, 0.065f);
    const float max_bridge_cross_gap = std::max(geometry.edge_length * 0.42f, 0.055f);
    bool emitted = false;

    for (uint32_t i = 0; i + 1 < pairs.size(); ++i) {
        const Vec3 a0 = pairs[i].a;
        const Vec3 a1 = pairs[i + 1].a;
        const Vec3 b0 = pairs[i].b;
        const Vec3 b1 = pairs[i + 1].b;

        if (pairs[i].fracture_chunk_id != pairs[i + 1].fracture_chunk_id) {
            continue;
        }

        const bool path_ok =
            length(a0 - a1) <= max_path_step &&
            length(b0 - b1) <= max_path_step &&
            pairs[i + 1].sort_value >= pairs[i].sort_value;
        const bool cross_ok =
            length(a0 - b0) <= max_bridge_cross_gap &&
            length(a1 - b1) <= max_bridge_cross_gap;
        const bool diagonals_ok =
            length(a1 - b0) <= max_triangle_edge &&
            length(a0 - b1) <= max_triangle_edge;

        if (!path_ok || !cross_ok || !diagonals_ok) {
            ++mesh.rejected_greedy_jump_count;
            continue;
        }

        const uint32_t material_id = pairs[i].material_id;
        const uint32_t fracture_chunk_id = pairs[i].fracture_chunk_id;
        append_chain_stitch_triangle(mesh, normalize(a0) * radius, normalize(b0) * radius, normalize(a1) * radius, cell_id, material_id, fracture_chunk_id);
        append_chain_stitch_triangle(mesh, normalize(a1) * radius, normalize(b0) * radius, normalize(b1) * radius, cell_id, material_id, fracture_chunk_id);
        ++mesh.greedy_path_step_count;
        emitted = true;
    }

    return emitted;
}

uint32_t shared_edge_plate_id(Vec3 direction, const MarchingCubesConfig& config) {
    if (!config.enable_fractures || !config.connect_fractures_across_cells) {
        return 0;
    }
    return nearest_global_fracture_chunk_id(direction, config);
}

Vec3 same_plate_edge_sample(
    const GoldbergCell& cell,
    const Frame& frame,
    Vec3 edge_direction,
    uint32_t chunk_id,
    float inset,
    float surface_radius,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    const Vec3 sample_direction = normalize(lerp(edge_direction, cell.center, inset));
    const Vec2 local = project_to_cell_plane(cell.center, frame, sample_direction);
    const uint32_t seed_count = global_fracture_seed_count(config);
    const Vec3 seed_direction = global_fracture_seed_position(chunk_id - 1u, seed_count, config);
    const Vec2 shard_center = project_to_cell_plane(cell.center, frame, seed_direction);
    const float top_radius = fracture_chunk_top_radius(surface_radius, chunk_id, config, cache);
    Vec2 adjusted = local;
    const Vec2 to_center = shard_center - local;
    const float to_center_length = length2(to_center);
    if (to_center_length > 0.000001f) {
        adjusted = adjusted + to_center * (std::min(config.fracture_gap, to_center_length * 0.45f) / to_center_length);
    }
    return fractured_sample_to_world(adjusted, cell.center, frame, top_radius);
}

bool shared_edge_interval_stays_on_plate(
    const GoldbergCell& cell_a,
    const GoldbergCell& cell_b,
    Vec3 edge0,
    Vec3 edge1,
    uint32_t chunk_id,
    float landing_inset,
    const MarchingCubesConfig& config
) {
    const Vec3 edge_mid = normalize(edge0 + edge1);
    const std::array<Vec3, 5> probes = {{
        edge0,
        edge_mid,
        edge1,
        normalize(lerp(edge_mid, cell_a.center, landing_inset)),
        normalize(lerp(edge_mid, cell_b.center, landing_inset)),
    }};

    for (Vec3 probe : probes) {
        if (shared_edge_plate_id(probe, config) != chunk_id) {
            return false;
        }
    }
    return true;
}

bool append_same_plate_shared_edge_fill(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const PointCloud& points,
    const BoundaryPairKey& pair_key,
    const BoundaryPairGeometry& geometry,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    if (!config.enable_fractures || !config.connect_fractures_across_cells) {
        return false;
    }

    const GoldbergCell& cell_a = topology.cells[pair_key.a];
    const GoldbergCell& cell_b = topology.cells[pair_key.b];
    const Frame frame_a = build_frame(cell_a.normal);
    const Frame frame_b = build_frame(cell_b.normal);
    const float surface_radius = (owned_surface_radius(points, pair_key.a) + owned_surface_radius(points, pair_key.b)) * 0.5f;
    constexpr uint32_t SamplesPerSharedEdge = 64;
    constexpr float LipInset = 0.0f;
    constexpr float LandingInset = 0.060f;
    bool emitted = false;

    for (uint32_t i = 0; i < SamplesPerSharedEdge; ++i) {
        const float t0 = static_cast<float>(i) / static_cast<float>(SamplesPerSharedEdge);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(SamplesPerSharedEdge);
        const float tm = (t0 + t1) * 0.5f;
        const Vec3 edge0 = normalize(lerp(geometry.edge_a, geometry.edge_b, t0));
        const Vec3 edge1 = normalize(lerp(geometry.edge_a, geometry.edge_b, t1));
        const Vec3 edge_mid = normalize(lerp(geometry.edge_a, geometry.edge_b, tm));
        const uint32_t chunk_id = shared_edge_plate_id(edge_mid, config);
        if (chunk_id == 0u || !shared_edge_interval_stays_on_plate(cell_a, cell_b, edge0, edge1, chunk_id, LandingInset, config)) {
            continue;
        }

        const Vec3 a_landing0 = same_plate_edge_sample(cell_a, frame_a, edge0, chunk_id, LandingInset, surface_radius, config, cache);
        const Vec3 a_landing1 = same_plate_edge_sample(cell_a, frame_a, edge1, chunk_id, LandingInset, surface_radius, config, cache);
        const Vec3 a_lip0 = same_plate_edge_sample(cell_a, frame_a, edge0, chunk_id, LipInset, surface_radius, config, cache);
        const Vec3 a_lip1 = same_plate_edge_sample(cell_a, frame_a, edge1, chunk_id, LipInset, surface_radius, config, cache);
        const Vec3 b_lip0 = same_plate_edge_sample(cell_b, frame_b, edge0, chunk_id, LipInset, surface_radius, config, cache);
        const Vec3 b_lip1 = same_plate_edge_sample(cell_b, frame_b, edge1, chunk_id, LipInset, surface_radius, config, cache);
        const Vec3 b_landing0 = same_plate_edge_sample(cell_b, frame_b, edge0, chunk_id, LandingInset, surface_radius, config, cache);
        const Vec3 b_landing1 = same_plate_edge_sample(cell_b, frame_b, edge1, chunk_id, LandingInset, surface_radius, config, cache);

        const uint32_t material_id = cell_a.kind == GoldbergCellKind::Pentagon ? 1u : 0u;
        append_chain_stitch_triangle(mesh, a_landing0, a_lip0, a_landing1, pair_key.a, material_id, chunk_id);
        append_chain_stitch_triangle(mesh, a_landing1, a_lip0, a_lip1, pair_key.a, material_id, chunk_id);
        append_chain_stitch_triangle(mesh, a_lip0, b_lip0, a_lip1, pair_key.a, material_id, chunk_id);
        append_chain_stitch_triangle(mesh, a_lip1, b_lip0, b_lip1, pair_key.a, material_id, chunk_id);
        append_chain_stitch_triangle(mesh, b_lip0, b_landing0, b_lip1, pair_key.a, material_id, chunk_id);
        append_chain_stitch_triangle(mesh, b_lip1, b_landing0, b_landing1, pair_key.a, material_id, chunk_id);
        mesh.greedy_path_step_count += 3;
        emitted = true;
    }

    return emitted;
}

bool append_same_plate_junction_cap(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const PointCloud& points,
    uint32_t corner_index,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    if (!config.enable_fractures || !config.connect_fractures_across_cells || corner_index >= topology.vertices.size()) {
        return false;
    }

    const Vec3 corner_direction = normalize(topology.vertices[corner_index].position);
    const uint32_t chunk_id = shared_edge_plate_id(corner_direction, config);
    if (chunk_id == 0u) {
        return false;
    }

    constexpr float JunctionInset = 0.070f;
    std::vector<JunctionCapPoint> cap_points;
    cap_points.reserve(3);
    float radius_sum = 0.0f;
    uint32_t radius_count = 0;
    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        if (std::find(cell.corner_indices.begin(), cell.corner_indices.end(), corner_index) == cell.corner_indices.end()) {
            continue;
        }

        const Vec3 sample_direction = normalize(lerp(corner_direction, cell.center, JunctionInset));
        if (shared_edge_plate_id(sample_direction, config) != chunk_id) {
            continue;
        }

        const Frame frame = build_frame(cell.normal);
        const float surface_radius = owned_surface_radius(points, cell_id);
        cap_points.push_back({
            same_plate_edge_sample(cell, frame, corner_direction, chunk_id, JunctionInset, surface_radius, config, cache),
            sample_direction,
            cell_id,
            0.0f,
        });
        radius_sum += surface_radius;
        ++radius_count;
    }

    if (cap_points.size() < 2 || radius_count == 0) {
        return false;
    }

    const Vec3 reference_axis = std::fabs(corner_direction.y) < 0.92f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference_axis, corner_direction));
    const Vec3 bitangent = cross(corner_direction, tangent);
    for (JunctionCapPoint& point : cap_points) {
        point.angle = std::atan2(dot(point.direction, bitangent), dot(point.direction, tangent));
    }
    std::sort(cap_points.begin(), cap_points.end(), [](const JunctionCapPoint& lhs, const JunctionCapPoint& rhs) {
        return lhs.angle < rhs.angle;
    });

    const float surface_radius = radius_sum / static_cast<float>(radius_count);
    const Vec3 center = corner_direction * fracture_chunk_top_radius(surface_radius, chunk_id, config, cache);
    const uint32_t material_id = topology.cells[cap_points.front().cell_id].kind == GoldbergCellKind::Pentagon ? 1u : 0u;
    bool emitted = false;

    for (uint32_t i = 0; i < cap_points.size(); ++i) {
        const JunctionCapPoint& a = cap_points[i];
        const JunctionCapPoint& b = cap_points[(i + 1u) % cap_points.size()];
        const Vec3 midpoint_direction = normalize(a.direction + b.direction);
        if (shared_edge_plate_id(midpoint_direction, config) != chunk_id) {
            continue;
        }

        Vec3 p0 = center;
        Vec3 p1 = a.position;
        Vec3 p2 = b.position;
        if (dot(cross(p1 - p0, p2 - p0), corner_direction) < 0.0f) {
            std::swap(p1, p2);
        }
        append_chain_stitch_triangle(mesh, p0, p1, p2, a.cell_id, material_id, chunk_id);
        ++mesh.greedy_path_step_count;
        emitted = true;
    }

    return emitted;
}

void append_same_plate_junction_caps(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const PointCloud& points,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    if (!config.enable_fractures || !config.connect_fractures_across_cells) {
        return;
    }

    for (uint32_t corner_index = 0; corner_index < topology.vertices.size(); ++corner_index) {
        append_same_plate_junction_cap(mesh, topology, points, corner_index, config, cache);
    }
}

void append_chain_stitches_for_pair(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const PointCloud& points,
    const BoundaryPairKey& pair_key,
    const BoundaryPairChains& chains,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    BoundaryPairGeometry geometry;
    if (!shared_boundary_geometry(topology, pair_key.a, pair_key.b, geometry)) {
        return;
    }

    const float radius = (owned_surface_radius(points, pair_key.a) + owned_surface_radius(points, pair_key.b)) * 0.5f;
    std::vector<CorridorPoint> side_a = collect_corridor_points(chains.a_segments, geometry, radius);
    std::vector<CorridorPoint> side_b = collect_corridor_points(chains.b_segments, geometry, radius);
    const bool emitted_same_plate_fill = append_same_plate_shared_edge_fill(mesh, topology, points, pair_key, geometry, config, cache);

    if (side_a.size() < 2 || side_b.size() < 2) {
        if (emitted_same_plate_fill) {
            ++mesh.shared_edge_path_count;
            ++mesh.boundary_run_count;
            ++mesh.paired_boundary_run_count;
            return;
        }
        ++mesh.unstitched_gap_count;
        return;
    }

    ++mesh.shared_edge_path_count;
    ++mesh.boundary_run_count;
    ++mesh.paired_boundary_run_count;
    if (!emitted_same_plate_fill) {
        append_greedy_corridor_stitches(
            mesh,
            side_a,
            side_b,
            geometry,
            pair_key.a,
            radius
        );
    }
}

void build_transition_stitches(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const PointCloud& points,
    const BoundaryEdgeMap& boundary_edges,
    const MarchingCubesConfig& config,
    const FractureBuildCache& cache
) {
    BoundaryPairMap boundary_pairs = build_boundary_pair_chains(mesh, topology, boundary_edges, config);

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        for (uint32_t neighbor_id : cell.neighbor_indices) {
            if (cell_id >= neighbor_id) {
                continue;
            }

            const BoundaryPairKey pair_key = boundary_pair_key(cell_id, neighbor_id);
            const auto found = boundary_pairs.find(pair_key);
            if (found == boundary_pairs.end()) {
                ++mesh.unstitched_gap_count;
                continue;
            }

            append_chain_stitches_for_pair(mesh, topology, points, pair_key, found->second, config, cache);
        }
    }

    append_same_plate_junction_caps(mesh, topology, points, config, cache);
}

void polygonize_cube(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    BoundaryEdgeMap& boundary_edges,
    uint32_t emitting_cell_id,
    const std::array<GridSample, 8>& cube,
    Vec3 center,
    const Frame& frame,
    const std::vector<Vec2>& clip_polygon,
    Vec3 quantize_step,
    bool quantize,
    float surface_radius,
    uint32_t material_id
) {
    constexpr std::array<std::array<uint32_t, 2>, 12> EdgeCorners = {{
        {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
        {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
        {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
    }};

    uint32_t cube_index = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        if (cube[i].density <= 0.0f) {
            cube_index |= (1u << i);
        }
    }

    const int edge_mask = mc_tables::EdgeTable[cube_index];
    if (edge_mask == 0) {
        return;
    }

    std::array<GridSample, 12> edge_vertices = {};
    for (uint32_t edge = 0; edge < EdgeCorners.size(); ++edge) {
        if ((edge_mask & (1 << edge)) != 0) {
            const auto corners = EdgeCorners[edge];
            edge_vertices[edge] = interpolate_projected(
                cube[corners[0]],
                cube[corners[1]],
                center,
                frame,
                quantize_step,
                quantize,
                surface_radius
            );
        }
    }

    for (uint32_t i = 0; i < 16 && mc_tables::TriTable[cube_index][i] != -1; i += 3) {
        const std::array<GridSample, 3> triangle = {{
            edge_vertices[mc_tables::TriTable[cube_index][i]],
            edge_vertices[mc_tables::TriTable[cube_index][i + 1]],
            edge_vertices[mc_tables::TriTable[cube_index][i + 2]],
        }};

        append_clipped_triangle(
            mesh,
            topology,
            boundary_edges,
            emitting_cell_id,
            triangle,
            center,
            frame,
            clip_polygon,
            surface_radius,
            material_id
        );
    }
}

VoxelKey voxel_key_for_position(Vec3 position, float grid_radius, float voxel_size, uint32_t resolution) {
    auto key_component = [&](float value) {
        const float normalized = (value + grid_radius) / voxel_size;
        const int32_t index = static_cast<int32_t>(std::floor(normalized));
        return static_cast<uint32_t>(std::clamp(index, 0, static_cast<int32_t>(resolution - 1u)));
    };
    return {
        key_component(position.x),
        key_component(position.y),
        key_component(position.z),
    };
}

Vec3 voxel_center_from_key(VoxelKey key, float grid_radius, float voxel_size) {
    return {
        -grid_radius + (static_cast<float>(key.x) + 0.5f) * voxel_size,
        -grid_radius + (static_cast<float>(key.y) + 0.5f) * voxel_size,
        -grid_radius + (static_cast<float>(key.z) + 0.5f) * voxel_size,
    };
}

std::vector<PreparedVoxelDig> prepare_voxel_digs(const VoxelEditSet& edits, float voxel_size) {
    std::vector<PreparedVoxelDig> prepared;
    prepared.reserve(edits.digs.size());
    const float leaf_radius = std::sqrt(3.0f) * voxel_size * 0.5f;
    for (const VoxelDigEdit& dig : edits.digs) {
        if (dig.radius_km <= 0.0f || !dig_target_matches(dig, VoxelDigTarget::Terrain)) {
            continue;
        }
        const float radius_mesh = kilometers_to_world_units(dig.radius_km);
        prepared.push_back({
            dig.center_mesh,
            radius_mesh,
            radius_mesh + leaf_radius,
        });
    }
    return prepared;
}

std::vector<PreparedVoxelDig> prepare_exact_voxel_digs(const VoxelEditSet& edits) {
    std::vector<PreparedVoxelDig> prepared;
    prepared.reserve(edits.digs.size());
    for (const VoxelDigEdit& dig : edits.digs) {
        if (dig.radius_km <= 0.0f || !dig_target_matches(dig, VoxelDigTarget::Terrain)) {
            continue;
        }
        const float radius_mesh = kilometers_to_world_units(dig.radius_km);
        prepared.push_back({
            dig.center_mesh,
            radius_mesh,
            radius_mesh,
        });
    }
    return prepared;
}

bool terrain_height_mask_project(const TerrainHeightMask& mask, Vec3 position, float& px, float& py) {
    if (mask.resolution == 0u ||
        mask.radius_km <= 0.0f ||
        mask.heights.size() != static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution) ||
        length(mask.center_mesh) <= 0.000001f ||
        length(position) <= 0.000001f) {
        return false;
    }

    const float radius_mesh = kilometers_to_world_units(mask.radius_km);
    const Vec3 relative = normalize(position) - normalize(mask.center_mesh);
    const float u = dot(relative, mask.tangent_mesh) / (radius_mesh * 2.0f) + 0.5f;
    const float v = dot(relative, mask.bitangent_mesh) / (radius_mesh * 2.0f) + 0.5f;
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return false;
    }

    px = u * static_cast<float>(mask.resolution - 1u);
    py = v * static_cast<float>(mask.resolution - 1u);
    return true;
}

bool terrain_height_mask_carves(const TerrainHeightMask& mask, Vec3 position) {
    float px = 0.0f;
    float py = 0.0f;
    if (!terrain_height_mask_project(mask, position, px, py)) {
        return false;
    }

    const uint32_t x = std::clamp(static_cast<uint32_t>(std::lround(px)), 0u, mask.resolution - 1u);
    const uint32_t y = std::clamp(static_cast<uint32_t>(std::lround(py)), 0u, mask.resolution - 1u);
    return mask.heights[static_cast<size_t>(y) * mask.resolution + x] == 0u;
}

bool terrain_height_masks_carve(Vec3 position, const std::vector<TerrainHeightMask>& masks) {
    for (const TerrainHeightMask& mask : masks) {
        if (terrain_height_mask_carves(mask, position)) {
            return true;
        }
    }
    return false;
}

bool terrain_height_mask_hole_footprint(const TerrainHeightMask& mask, Vec3& center, float& radius) {
    center = {};
    radius = 0.0f;
    if (mask.resolution == 0u ||
        mask.radius_km <= 0.0f ||
        mask.heights.size() != static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution) ||
        length(mask.center_mesh) <= 0.000001f) {
        return false;
    }

    uint32_t min_x = mask.resolution;
    uint32_t min_y = mask.resolution;
    uint32_t max_x = 0u;
    uint32_t max_y = 0u;
    bool any_hole = false;
    for (uint32_t y = 0; y < mask.resolution; ++y) {
        for (uint32_t x = 0; x < mask.resolution; ++x) {
            if (mask.heights[static_cast<size_t>(y) * mask.resolution + x] != 0u) {
                continue;
            }
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            any_hole = true;
        }
    }
    if (!any_hole) {
        return false;
    }

    const float resolution_minus_one = static_cast<float>(std::max(1u, mask.resolution - 1u));
    auto local_coord = [&](uint32_t value) {
        return ((static_cast<float>(value) / resolution_minus_one) - 0.5f) * kilometers_to_world_units(mask.radius_km) * 2.0f;
    };
    const float min_local_x = local_coord(min_x);
    const float max_local_x = local_coord(max_x);
    const float min_local_y = local_coord(min_y);
    const float max_local_y = local_coord(max_y);
    const Vec3 local_center =
        mask.tangent_mesh * ((min_local_x + max_local_x) * 0.5f) +
        mask.bitangent_mesh * ((min_local_y + max_local_y) * 0.5f);
    center = normalize(normalize(mask.center_mesh) + local_center);
    radius = length(mask.tangent_mesh * ((max_local_x - min_local_x) * 0.5f) +
                    mask.bitangent_mesh * ((max_local_y - min_local_y) * 0.5f));
    return radius > 0.0f;
}

uint32_t apply_voxel_dig_edits(
    std::vector<VoxelKey>& keys,
    float grid_radius,
    float voxel_size,
    const VoxelEditSet& edits
) {
    if (keys.empty() || edits.digs.empty()) {
        return 0;
    }

    const std::vector<PreparedVoxelDig> digs = prepare_voxel_digs(edits, voxel_size);
    if (digs.empty()) {
        return 0;
    }

    const size_t original_size = keys.size();
    keys.erase(
        std::remove_if(
            keys.begin(),
            keys.end(),
            [&](VoxelKey key) {
                const Vec3 center = voxel_center_from_key(key, grid_radius, voxel_size);
                for (const PreparedVoxelDig& dig : digs) {
                    const float distance_to_dig = length(center - dig.center_mesh);
                    if (distance_to_dig <= dig.radius_with_leaf_mesh) {
                        return true;
                    }
                }
                return false;
            }
        ),
        keys.end()
    );

    return static_cast<uint32_t>(original_size - keys.size());
}

uint32_t apply_exact_voxel_dig_edits(
    std::vector<VoxelKey>& keys,
    float grid_radius,
    float voxel_size,
    const VoxelEditSet& edits
) {
    if (keys.empty()) {
        return 0;
    }

    const std::vector<PreparedVoxelDig> digs = prepare_exact_voxel_digs(edits);
    if (digs.empty() && edits.terrain_masks.empty()) {
        return 0;
    }

    const size_t original_size = keys.size();
    keys.erase(
        std::remove_if(
            keys.begin(),
            keys.end(),
            [&](VoxelKey key) {
                const Vec3 center = voxel_center_from_key(key, grid_radius, voxel_size);
                for (const PreparedVoxelDig& dig : digs) {
                    if (length(center - dig.center_mesh) <= dig.radius_mesh) {
                        return true;
                    }
                }
                if (terrain_height_masks_carve(center, edits.terrain_masks)) {
                    return true;
                }
                return false;
            }
        ),
        keys.end()
    );

    return static_cast<uint32_t>(original_size - keys.size());
}

uint32_t child_index_for_voxel(VoxelKey key, uint32_t origin_x, uint32_t origin_y, uint32_t origin_z, uint32_t child_size) {
    const uint32_t x_bit = key.x >= origin_x + child_size ? 4u : 0u;
    const uint32_t y_bit = key.y >= origin_y + child_size ? 2u : 0u;
    const uint32_t z_bit = key.z >= origin_z + child_size ? 1u : 0u;
    return x_bit | y_bit | z_bit;
}

uint32_t child_origin_x(uint32_t origin, uint32_t child_index, uint32_t child_size) {
    return origin + ((child_index & 4u) != 0u ? child_size : 0u);
}

uint32_t child_origin_y(uint32_t origin, uint32_t child_index, uint32_t child_size) {
    return origin + ((child_index & 2u) != 0u ? child_size : 0u);
}

uint32_t child_origin_z(uint32_t origin, uint32_t child_index, uint32_t child_size) {
    return origin + ((child_index & 1u) != 0u ? child_size : 0u);
}

uint64_t voxelized_triangle_sample_count(
    const QuantizedMesh& mesh,
    const std::vector<uint32_t>& indices,
    float voxel_size
) {
    uint64_t count = 0;
    for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vec3 a = mesh.vertices[indices[i]].position;
        const Vec3 b = mesh.vertices[indices[i + 1]].position;
        const Vec3 c = mesh.vertices[indices[i + 2]].position;
        const float longest_edge = std::max(length(b - a), std::max(length(c - b), length(a - c)));
        const float sample_spacing = std::max(voxel_size * 2.0f, 0.000001f);
        const uint32_t steps = std::clamp(static_cast<uint32_t>(std::ceil(longest_edge / sample_spacing)), 1u, 256u);
        count += (static_cast<uint64_t>(steps) + 1ull) * (static_cast<uint64_t>(steps) + 2ull) / 2ull + 4ull;
    }
    return count;
}

void add_voxelized_triangles(
    std::vector<VoxelKey>& keys,
    const QuantizedMesh& mesh,
    const std::vector<uint32_t>& indices,
    float grid_radius,
    float voxel_size,
    uint32_t resolution,
    const MarchingCubesConfig& config,
    float progress_begin,
    float progress_end,
    uint64_t& processed_vertices,
    uint64_t total_vertices
) {
    auto push_voxel_sample = [&](Vec3 position) {
        keys.push_back(voxel_key_for_position(position, grid_radius, voxel_size, resolution));
        ++processed_vertices;
        if (total_vertices > 0u) {
            const double t = static_cast<double>(processed_vertices) / static_cast<double>(total_vertices);
            report_progress(config, static_cast<double>(progress_begin) + static_cast<double>(progress_end - progress_begin) * t, nullptr);
        }
    };

    for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vec3 a = mesh.vertices[indices[i]].position;
        const Vec3 b = mesh.vertices[indices[i + 1]].position;
        const Vec3 c = mesh.vertices[indices[i + 2]].position;
        const float longest_edge = std::max(length(b - a), std::max(length(c - b), length(a - c)));
        const float sample_spacing = std::max(voxel_size * 2.0f, 0.000001f);
        const uint32_t steps = std::clamp(static_cast<uint32_t>(std::ceil(longest_edge / sample_spacing)), 1u, 256u);
        for (uint32_t u_step = 0; u_step <= steps; ++u_step) {
            for (uint32_t v_step = 0; v_step + u_step <= steps; ++v_step) {
                const float u = static_cast<float>(u_step) / static_cast<float>(steps);
                const float v = static_cast<float>(v_step) / static_cast<float>(steps);
                const float w = 1.0f - u - v;
                push_voxel_sample(a * u + b * v + c * w);
            }
        }
        push_voxel_sample((a + b + c) / 3.0f);
        push_voxel_sample((a + b) * 0.5f);
        push_voxel_sample((b + c) * 0.5f);
        push_voxel_sample((c + a) * 0.5f);
    }
}

uint32_t surface_net_grid_index(uint32_t x, uint32_t y, uint32_t z, uint32_t resolution) {
    return (z * resolution + y) * resolution + x;
}

uint32_t surface_net_cube_index(uint32_t x, uint32_t y, uint32_t z, uint32_t cube_resolution) {
    return (z * cube_resolution + y) * cube_resolution + x;
}

bool surface_net_bit(const std::vector<uint64_t>& bits, uint32_t index) {
    return (bits[index / 64u] & (uint64_t{1} << (index % 64u))) != 0u;
}

void set_surface_net_bit(std::vector<uint64_t>& bits, uint32_t index) {
    bits[index / 64u] |= (uint64_t{1} << (index % 64u));
}

void set_surface_net_bit64(std::vector<uint64_t>& bits, uint64_t index) {
    bits[static_cast<size_t>(index / 64u)] |= (uint64_t{1} << (index % 64u));
}

bool set_surface_net_bit64_if_unset(std::vector<uint64_t>& bits, uint64_t index) {
    uint64_t& word = bits[static_cast<size_t>(index / 64u)];
    const uint64_t mask = uint64_t{1} << (index % 64u);
    if ((word & mask) != 0u) {
        return false;
    }
    word |= mask;
    return true;
}

uint32_t surface_net_bit_pair(const std::vector<uint64_t>& bits, uint64_t index) {
    const size_t word_index = static_cast<size_t>(index >> 6u);
    const uint32_t shift = static_cast<uint32_t>(index & 63u);
    if (shift < 63u) {
        return static_cast<uint32_t>((bits[word_index] >> shift) & 3ull);
    }
    const uint32_t low = static_cast<uint32_t>((bits[word_index] >> 63u) & 1ull);
    const uint32_t high = static_cast<uint32_t>(bits[word_index + 1u] & 1ull);
    return low | (high << 1u);
}

bool set_surface_net_cube_candidate(std::vector<uint64_t>& bits, uint32_t cube_resolution, uint32_t x, uint32_t y, uint32_t z) {
    if (x >= cube_resolution || y >= cube_resolution || z >= cube_resolution) {
        return false;
    }
    return set_surface_net_bit64_if_unset(bits, surface_net_cube_index(x, y, z, cube_resolution));
}

uint64_t surface_net_edge_index(uint32_t x, uint32_t y, uint32_t z, uint32_t axis, uint32_t resolution) {
    return static_cast<uint64_t>(axis) * static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) +
           static_cast<uint64_t>(surface_net_grid_index(x, y, z, resolution));
}

bool set_surface_net_edge_candidate_bit(
    std::vector<uint64_t>& edges,
    const std::vector<uint64_t>& occupancy,
    uint32_t resolution,
    uint32_t x,
    uint32_t y,
    uint32_t z,
    uint32_t axis,
    bool negative_direction,
    SurfaceNetEdgeKey* edge_key = nullptr
) {
    auto set_edge = [&](uint32_t edge_x, uint32_t edge_y, uint32_t edge_z) {
        const uint64_t edge_index = surface_net_edge_index(edge_x, edge_y, edge_z, axis, resolution);
        if (!set_surface_net_bit64_if_unset(edges, edge_index)) {
            return false;
        }
        if (edge_key) {
            *edge_key = {edge_x, edge_y, edge_z, axis};
        }
        return true;
    };

    if (axis == 0u) {
        if (negative_direction) {
            if (x == 0u) return false;
            if (!surface_net_bit(occupancy, surface_net_grid_index(x - 1u, y, z, resolution))) {
                return set_edge(x - 1u, y, z);
            }
        } else if (x + 1u < resolution && !surface_net_bit(occupancy, surface_net_grid_index(x + 1u, y, z, resolution))) {
            return set_edge(x, y, z);
        }
    } else if (axis == 1u) {
        if (negative_direction) {
            if (y == 0u) return false;
            if (!surface_net_bit(occupancy, surface_net_grid_index(x, y - 1u, z, resolution))) {
                return set_edge(x, y - 1u, z);
            }
        } else if (y + 1u < resolution && !surface_net_bit(occupancy, surface_net_grid_index(x, y + 1u, z, resolution))) {
            return set_edge(x, y, z);
        }
    } else {
        if (negative_direction) {
            if (z == 0u) return false;
            if (!surface_net_bit(occupancy, surface_net_grid_index(x, y, z - 1u, resolution))) {
                return set_edge(x, y, z - 1u);
            }
        } else if (z + 1u < resolution && !surface_net_bit(occupancy, surface_net_grid_index(x, y, z + 1u, resolution))) {
            return set_edge(x, y, z);
        }
    }
    return false;
}

size_t count_set_bits(const std::vector<uint64_t>& bits) {
    size_t count = 0;
    for (uint64_t word : bits) {
        count += static_cast<size_t>(std::popcount(word));
    }
    return count;
}

std::vector<VoxelKey> compact_surface_net_cube_candidates(
    const std::vector<uint64_t>& bits,
    uint32_t cube_resolution,
    const MarchingCubesConfig& config,
    double progress_begin,
    double progress_end
) {
    std::vector<VoxelKey> candidates;
    const uint64_t cube_count = static_cast<uint64_t>(cube_resolution) * static_cast<uint64_t>(cube_resolution) * static_cast<uint64_t>(cube_resolution);
    candidates.reserve(std::min<size_t>(count_set_bits(bits), static_cast<size_t>(cube_count)));
    const uint64_t plane = static_cast<uint64_t>(cube_resolution) * static_cast<uint64_t>(cube_resolution);
    for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
        if ((word_index & 4095u) == 0u) {
            report_index_progress(config, progress_begin, progress_end, static_cast<uint64_t>(word_index), static_cast<uint64_t>(bits.size()), nullptr);
        }
        uint64_t word = bits[word_index];
        while (word != 0u) {
            const uint32_t bit = static_cast<uint32_t>(std::countr_zero(word));
            const uint64_t index = static_cast<uint64_t>(word_index) * 64ull + bit;
            if (index < cube_count) {
                const uint32_t z = static_cast<uint32_t>(index / plane);
                const uint64_t remainder = index - static_cast<uint64_t>(z) * plane;
                const uint32_t y = static_cast<uint32_t>(remainder / cube_resolution);
                const uint32_t x = static_cast<uint32_t>(remainder - static_cast<uint64_t>(y) * cube_resolution);
                candidates.push_back({x, y, z});
            }
            word &= word - 1u;
        }
    }
    report_progress(config, progress_end, nullptr);
    return candidates;
}

std::vector<SurfaceNetEdgeKey> compact_surface_net_edge_candidates(
    const std::vector<uint64_t>& bits,
    uint32_t resolution,
    const MarchingCubesConfig& config,
    double progress_begin,
    double progress_end
) {
    std::vector<SurfaceNetEdgeKey> candidates;
    const uint64_t grid_count = static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution);
    const uint64_t total_count = grid_count * 3ull;
    candidates.reserve(std::min<size_t>(count_set_bits(bits), static_cast<size_t>(total_count)));
    const uint64_t plane = static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution);
    for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
        if ((word_index & 4095u) == 0u) {
            report_index_progress(config, progress_begin, progress_end, static_cast<uint64_t>(word_index), static_cast<uint64_t>(bits.size()), nullptr);
        }
        uint64_t word = bits[word_index];
        while (word != 0u) {
            const uint32_t bit = static_cast<uint32_t>(std::countr_zero(word));
            const uint64_t index = static_cast<uint64_t>(word_index) * 64ull + bit;
            if (index < total_count) {
                const uint32_t axis = static_cast<uint32_t>(index / grid_count);
                const uint64_t grid_index = index - static_cast<uint64_t>(axis) * grid_count;
                const uint32_t z = static_cast<uint32_t>(grid_index / plane);
                const uint64_t remainder = grid_index - static_cast<uint64_t>(z) * plane;
                const uint32_t y = static_cast<uint32_t>(remainder / resolution);
                const uint32_t x = static_cast<uint32_t>(remainder - static_cast<uint64_t>(y) * resolution);
                candidates.push_back({x, y, z, axis});
            }
            word &= word - 1u;
        }
    }
    report_progress(config, progress_end, nullptr);
    return candidates;
}

int32_t surface_net_cube_vertex(
    const std::vector<int32_t>& cube_vertices,
    uint32_t cube_resolution,
    uint32_t x,
    uint32_t y,
    uint32_t z
);

void append_surface_net_quad(
    SurfaceNetMesh& surface_net,
    int32_t a,
    int32_t b,
    int32_t c,
    int32_t d
);

bool write_surface_net_edge_candidate_quad(
    const std::vector<int32_t>& cube_vertices,
    uint32_t cube_resolution,
    SurfaceNetEdgeKey edge,
    uint32_t* output
) {
    int32_t a = -1;
    int32_t b = -1;
    int32_t c = -1;
    int32_t d = -1;
    if (edge.axis == 0u) {
        if (edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
            return false;
        }
        a = cube_vertices[surface_net_cube_index(edge.x, edge.y - 1u, edge.z - 1u, cube_resolution)];
        b = cube_vertices[surface_net_cube_index(edge.x, edge.y, edge.z - 1u, cube_resolution)];
        c = cube_vertices[surface_net_cube_index(edge.x, edge.y, edge.z, cube_resolution)];
        d = cube_vertices[surface_net_cube_index(edge.x, edge.y - 1u, edge.z, cube_resolution)];
    } else if (edge.axis == 1u) {
        if (edge.y >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
            return false;
        }
        a = cube_vertices[surface_net_cube_index(edge.x - 1u, edge.y, edge.z - 1u, cube_resolution)];
        b = cube_vertices[surface_net_cube_index(edge.x, edge.y, edge.z - 1u, cube_resolution)];
        c = cube_vertices[surface_net_cube_index(edge.x, edge.y, edge.z, cube_resolution)];
        d = cube_vertices[surface_net_cube_index(edge.x - 1u, edge.y, edge.z, cube_resolution)];
    } else {
        if (edge.z >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution) {
            return false;
        }
        a = cube_vertices[surface_net_cube_index(edge.x - 1u, edge.y - 1u, edge.z, cube_resolution)];
        b = cube_vertices[surface_net_cube_index(edge.x, edge.y - 1u, edge.z, cube_resolution)];
        c = cube_vertices[surface_net_cube_index(edge.x, edge.y, edge.z, cube_resolution)];
        d = cube_vertices[surface_net_cube_index(edge.x - 1u, edge.y, edge.z, cube_resolution)];
    }
    if (a < 0 || b < 0 || c < 0 || d < 0) {
        return false;
    }
    output[0] = static_cast<uint32_t>(a);
    output[1] = static_cast<uint32_t>(b);
    output[2] = static_cast<uint32_t>(c);
    output[3] = static_cast<uint32_t>(a);
    output[4] = static_cast<uint32_t>(c);
    output[5] = static_cast<uint32_t>(d);
    return true;
}

void append_surface_net_edge_candidate_quad(
    SurfaceNetMesh& surface_net,
    const std::vector<int32_t>& cube_vertices,
    uint32_t cube_resolution,
    SurfaceNetEdgeKey edge
) {
    uint32_t indices[6] = {};
    if (!write_surface_net_edge_candidate_quad(cube_vertices, cube_resolution, edge, indices)) {
        return;
    }
    surface_net.triangle_indices.insert(surface_net.triangle_indices.end(), std::begin(indices), std::end(indices));
}

void append_surface_net_quads_from_edge_bits(
    SurfaceNetMesh& surface_net,
    const std::vector<uint64_t>& bits,
    const std::vector<int32_t>& cube_vertices,
    uint32_t resolution,
    uint32_t cube_resolution
) {
    const uint64_t grid_count = static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution);
    const uint64_t total_count = grid_count * 3ull;
    const uint64_t plane = static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution);
    for (size_t word_index = 0; word_index < bits.size(); ++word_index) {
        uint64_t word = bits[word_index];
        while (word != 0u) {
            const uint32_t bit = static_cast<uint32_t>(std::countr_zero(word));
            const uint64_t index = static_cast<uint64_t>(word_index) * 64ull + bit;
            if (index < total_count) {
                const uint32_t axis = static_cast<uint32_t>(index / grid_count);
                const uint64_t grid_index = index - static_cast<uint64_t>(axis) * grid_count;
                const uint32_t z = static_cast<uint32_t>(grid_index / plane);
                const uint64_t remainder = grid_index - static_cast<uint64_t>(z) * plane;
                const uint32_t y = static_cast<uint32_t>(remainder / resolution);
                const uint32_t x = static_cast<uint32_t>(remainder - static_cast<uint64_t>(y) * resolution);
                append_surface_net_edge_candidate_quad(surface_net, cube_vertices, cube_resolution, {x, y, z, axis});
            }
            word &= word - 1u;
        }
    }
}

Vec3 surface_net_sample_position(uint32_t x, uint32_t y, uint32_t z, float grid_radius, float voxel_size) {
    return {
        -grid_radius + (static_cast<float>(x) + 0.5f) * voxel_size,
        -grid_radius + (static_cast<float>(y) + 0.5f) * voxel_size,
        -grid_radius + (static_cast<float>(z) + 0.5f) * voxel_size,
    };
}

std::vector<VoxelKey> coarsen_surface_net_keys(const std::vector<VoxelKey>& keys, uint32_t source_depth, uint32_t target_depth) {
    std::vector<VoxelKey> result;
    result.reserve(keys.size());
    const uint32_t shift = source_depth > target_depth ? source_depth - target_depth : 0u;
    for (VoxelKey key : keys) {
        if (shift > 0u) {
            key.x >>= shift;
            key.y >>= shift;
            key.z >>= shift;
        }
        result.push_back(key);
    }
    sort_and_unique_voxel_keys(result);
    return result;
}

Vec3 voxel_min_from_key(VoxelKey key, float grid_radius, float voxel_size) {
    return {
        -grid_radius + static_cast<float>(key.x) * voxel_size,
        -grid_radius + static_cast<float>(key.y) * voxel_size,
        -grid_radius + static_cast<float>(key.z) * voxel_size,
    };
}

Vec3 voxel_max_from_key(VoxelKey key, float grid_radius, float voxel_size) {
    const Vec3 minimum = voxel_min_from_key(key, grid_radius, voxel_size);
    return minimum + Vec3{voxel_size, voxel_size, voxel_size};
}

bool aabb_intersects_sphere(Vec3 minimum, Vec3 maximum, Vec3 center, float radius) {
    auto axis_distance_sq = [](float value, float minimum_axis, float maximum_axis) {
        if (value < minimum_axis) {
            const float distance = minimum_axis - value;
            return distance * distance;
        }
        if (value > maximum_axis) {
            const float distance = value - maximum_axis;
            return distance * distance;
        }
        return 0.0f;
    };
    const float distance_sq =
        axis_distance_sq(center.x, minimum.x, maximum.x) +
        axis_distance_sq(center.y, minimum.y, maximum.y) +
        axis_distance_sq(center.z, minimum.z, maximum.z);
    return distance_sq <= radius * radius;
}

std::vector<VoxelKey> promoted_depth8_keys_for_dig(
    Vec3 center,
    float radius,
    float grid_radius
) {
    std::vector<VoxelKey> keys;
    if (radius <= 0.0f || grid_radius <= 0.0f) {
        return keys;
    }

    const uint32_t root_resolution = 1u << SurfacePromotionRootDepth;
    const float root_voxel_size = (grid_radius * 2.0f) / static_cast<float>(root_resolution);
    const float promotion_radius = radius + root_voxel_size;
    auto clamped_key_component = [&](float value) {
        const float normalized = (value + grid_radius) / root_voxel_size;
        const int32_t index = static_cast<int32_t>(std::floor(normalized));
        return static_cast<uint32_t>(std::clamp(index, 0, static_cast<int32_t>(root_resolution - 1u)));
    };

    const uint32_t min_x = clamped_key_component(center.x - promotion_radius);
    const uint32_t min_y = clamped_key_component(center.y - promotion_radius);
    const uint32_t min_z = clamped_key_component(center.z - promotion_radius);
    const uint32_t max_x = clamped_key_component(center.x + promotion_radius);
    const uint32_t max_y = clamped_key_component(center.y + promotion_radius);
    const uint32_t max_z = clamped_key_component(center.z + promotion_radius);
    keys.reserve(static_cast<size_t>(max_x - min_x + 1u) * (max_y - min_y + 1u) * (max_z - min_z + 1u));

    for (uint32_t z = min_z; z <= max_z; ++z) {
        for (uint32_t y = min_y; y <= max_y; ++y) {
            for (uint32_t x = min_x; x <= max_x; ++x) {
                const VoxelKey key{x, y, z};
                if (aabb_intersects_sphere(
                        voxel_min_from_key(key, grid_radius, root_voxel_size),
                        voxel_max_from_key(key, grid_radius, root_voxel_size),
                        center,
                        promotion_radius)) {
                    keys.push_back(key);
                }
            }
        }
    }

    sort_and_unique_voxel_keys(keys);
    return keys;
}

void update_patch_replacement_bounds(LocalSurfaceNetPatch& patch, float grid_radius) {
    if (patch.promoted_depth8_keys.empty() || grid_radius <= 0.0f) {
        const Vec3 extent{patch.extraction_radius_mesh, patch.extraction_radius_mesh, patch.extraction_radius_mesh};
        patch.replacement_min = patch.center_mesh - extent;
        patch.replacement_max = patch.center_mesh + extent;
        return;
    }

    const float root_voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << SurfacePromotionRootDepth);
    Vec3 minimum{1000000.0f, 1000000.0f, 1000000.0f};
    Vec3 maximum{-1000000.0f, -1000000.0f, -1000000.0f};
    for (VoxelKey key : patch.promoted_depth8_keys) {
        const Vec3 key_min = voxel_min_from_key(key, grid_radius, root_voxel_size);
        const Vec3 key_max = voxel_max_from_key(key, grid_radius, root_voxel_size);
        minimum.x = std::min(minimum.x, key_min.x);
        minimum.y = std::min(minimum.y, key_min.y);
        minimum.z = std::min(minimum.z, key_min.z);
        maximum.x = std::max(maximum.x, key_max.x);
        maximum.y = std::max(maximum.y, key_max.y);
        maximum.z = std::max(maximum.z, key_max.z);
    }
    patch.replacement_min = minimum;
    patch.replacement_max = maximum;
    patch.center_mesh = (minimum + maximum) * 0.5f;
    patch.extraction_radius_mesh = length((maximum - minimum) * 0.5f);
    patch.suppress_radius_mesh = patch.extraction_radius_mesh;
}

void expand_patch_replacement_bounds(LocalSurfaceNetPatch& patch, float amount, float grid_radius) {
    if (amount <= 0.0f || grid_radius <= 0.0f) {
        return;
    }

    patch.replacement_min = {
        std::max(-grid_radius, patch.replacement_min.x - amount),
        std::max(-grid_radius, patch.replacement_min.y - amount),
        std::max(-grid_radius, patch.replacement_min.z - amount),
    };
    patch.replacement_max = {
        std::min(grid_radius, patch.replacement_max.x + amount),
        std::min(grid_radius, patch.replacement_max.y + amount),
        std::min(grid_radius, patch.replacement_max.z + amount),
    };
    patch.center_mesh = (patch.replacement_min + patch.replacement_max) * 0.5f;
    patch.extraction_radius_mesh = length((patch.replacement_max - patch.replacement_min) * 0.5f);
    patch.suppress_radius_mesh = patch.extraction_radius_mesh;
}

bool depth8_key_in_promoted_patches(VoxelKey key, const std::vector<LocalSurfaceNetPatch>& patches) {
    for (const LocalSurfaceNetPatch& patch : patches) {
        if (!patch.replace_surface || patch.promoted_depth8_keys.empty()) {
            continue;
        }
        if (std::binary_search(patch.promoted_depth8_keys.begin(), patch.promoted_depth8_keys.end(), key)) {
            return true;
        }
    }
    return false;
}

VoxelEditSet promoted_terrain_dig_edits(const MarchingCubesConfig& config) {
    VoxelEditSet promoted;
    promoted.local_depth = config.voxel_edits.local_depth;
    const float promotion_radius_mesh = kilometers_to_world_units(std::max(0.0f, config.terrain_svo_promotion_radius_km));
    if (promotion_radius_mesh <= 0.0f) {
        return promoted;
    }

    struct TerrainDigRef {
        Vec3 center;
        float radius = 0.0f;
        uint32_t depth = MaxSvoDepth;
        size_t mask_index = std::numeric_limits<size_t>::max();
    };

    std::vector<TerrainDigRef> terrain_digs;
    terrain_digs.reserve(config.voxel_edits.digs.size() + config.voxel_edits.terrain_masks.size());
    for (const VoxelDigEdit& dig : config.voxel_edits.digs) {
        if (!dig_target_matches(dig, VoxelDigTarget::Terrain) || dig.radius_km <= 0.0f || length(dig.center_mesh) <= 0.000001f) {
            continue;
        }
        terrain_digs.push_back({
            dig.center_mesh,
            kilometers_to_world_units(dig.radius_km),
            std::clamp(dig.depth > 0u ? dig.depth : config.voxel_edits.local_depth, 1u, MaxSvoDepth),
            std::numeric_limits<size_t>::max(),
        });
    }
    for (size_t mask_index = 0; mask_index < config.voxel_edits.terrain_masks.size(); ++mask_index) {
        Vec3 center{};
        float radius = 0.0f;
        if (!terrain_height_mask_hole_footprint(config.voxel_edits.terrain_masks[mask_index], center, radius)) {
            continue;
        }
        terrain_digs.push_back({
            center,
            radius,
            std::clamp(config.voxel_edits.local_depth, 1u, MaxSvoDepth),
            mask_index,
        });
    }
    if (terrain_digs.empty()) {
        return promoted;
    }

    std::vector<uint8_t> visited(terrain_digs.size(), 0u);
    std::vector<size_t> stack;
    std::vector<size_t> cluster;
    for (size_t start = 0; start < terrain_digs.size(); ++start) {
        if (visited[start] != 0u) {
            continue;
        }

        stack.clear();
        cluster.clear();
        visited[start] = 1u;
        stack.push_back(start);
        while (!stack.empty()) {
            const size_t current = stack.back();
            stack.pop_back();
            cluster.push_back(current);
            for (size_t candidate = 0; candidate < terrain_digs.size(); ++candidate) {
                if (visited[candidate] != 0u) {
                    continue;
                }
                const float distance = length(terrain_digs[current].center - terrain_digs[candidate].center);
                if (distance <= terrain_digs[current].radius + terrain_digs[candidate].radius) {
                    visited[candidate] = 1u;
                    stack.push_back(candidate);
                }
            }
        }

        Vec3 center_sum{};
        float weight_sum = 0.0f;
        uint32_t depth = 1u;
        bool cluster_has_circular_dig = false;
        for (size_t index : cluster) {
            const float weight = std::max(terrain_digs[index].radius, 0.000001f);
            center_sum = center_sum + terrain_digs[index].center * weight;
            weight_sum += weight;
            depth = std::max(depth, terrain_digs[index].depth);
            cluster_has_circular_dig = cluster_has_circular_dig ||
                terrain_digs[index].mask_index == std::numeric_limits<size_t>::max();
        }
        if (weight_sum <= 0.0f || length(center_sum) <= 0.000001f) {
            continue;
        }

        const Vec3 cluster_center = center_sum * (1.0f / weight_sum);
        float cluster_radius = 0.0f;
        for (size_t index : cluster) {
            cluster_radius = std::max(cluster_radius, length(terrain_digs[index].center - cluster_center) + terrain_digs[index].radius);
        }
        if (cluster_radius < promotion_radius_mesh) {
            continue;
        }

        if (cluster_has_circular_dig) {
            promoted.digs.push_back({
                cluster_center,
                world_units_to_kilometers(cluster_radius),
                depth,
                VoxelDigTarget::Terrain,
            });
        }
        for (size_t index : cluster) {
            const size_t mask_index = terrain_digs[index].mask_index;
            if (mask_index == std::numeric_limits<size_t>::max()) {
                continue;
            }
            const TerrainHeightMask& mask = config.voxel_edits.terrain_masks[mask_index];
            const bool already_added = std::any_of(
                promoted.terrain_masks.begin(),
                promoted.terrain_masks.end(),
                [&](const TerrainHeightMask& existing) {
                    return existing.revision == mask.revision &&
                           length(existing.center_mesh - mask.center_mesh) <= 0.000001f;
                }
            );
            if (!already_added) {
                promoted.terrain_masks.push_back(mask);
            }
        }
    }

    return promoted;
}

std::vector<LocalSurfaceNetPatch> build_local_surface_net_patches(const MarchingCubesConfig& config, float grid_radius) {
    std::vector<LocalSurfaceNetPatch> patches;
    if (!config.enable_local_surface_net_detail || config.local_surface_net_max_patches == 0u) {
        return patches;
    }

    const float overlap = kilometers_to_world_units(std::max(0.0f, config.local_surface_net_patch_overlap_km));
    const uint32_t fallback_depth = std::clamp(
        config.local_surface_net_depth > 0u ? config.local_surface_net_depth : config.voxel_edits.local_depth,
        1u,
        MaxSvoDepth
    );
    auto add_patch = [&](Vec3 center, float dig_radius_mesh, uint32_t requested_depth, bool replace_surface) {
        if (patches.size() >= config.local_surface_net_max_patches || length(center) <= 0.000001f) {
            return;
        }
        const float minimum_radius = kilometers_to_world_units(4.0f);
        const float surface_pad = kilometers_to_world_units(1.5f);
        const float suppress_radius = std::max(dig_radius_mesh + surface_pad, minimum_radius);
        const float detail_radius = std::max(
            kilometers_to_world_units(config.local_surface_net_patch_radius_km),
            dig_radius_mesh + std::max(overlap, kilometers_to_world_units(48.0f))
        );
        const float extraction_radius = std::max(suppress_radius, detail_radius);
        const uint32_t depth = std::clamp(requested_depth > 0u ? requested_depth : fallback_depth, 1u, MaxSvoDepth);
        std::vector<VoxelKey> promoted_keys = replace_surface
            ? promoted_depth8_keys_for_dig(center, dig_radius_mesh, grid_radius)
            : std::vector<VoxelKey>{};
        if (!promoted_keys.empty()) {
            promoted_keys.erase(
                std::remove_if(
                    promoted_keys.begin(),
                    promoted_keys.end(),
                    [&](VoxelKey key) {
                        return std::any_of(
                            patches.begin(),
                            patches.end(),
                            [&](const LocalSurfaceNetPatch& patch) {
                                return std::binary_search(
                                    patch.promoted_depth8_keys.begin(),
                                    patch.promoted_depth8_keys.end(),
                                    key
                                );
                            }
                        );
                    }
                ),
                promoted_keys.end()
            );
        }
        if (replace_surface && promoted_keys.empty()) {
            return;
        }
        for (const LocalSurfaceNetPatch& patch : patches) {
            if (!replace_surface && length(patch.center_mesh - center) < suppress_radius * 0.5f) {
                return;
            }
        }
        LocalSurfaceNetPatch patch;
        patch.center_mesh = center;
        patch.dig_center_mesh = center;
        patch.dig_radius_mesh = dig_radius_mesh;
        patch.suppress_radius_mesh = suppress_radius;
        patch.extraction_radius_mesh = extraction_radius;
        patch.depth = depth;
        patch.replace_surface = replace_surface;
        patch.promoted_depth8_keys = std::move(promoted_keys);
        update_patch_replacement_bounds(patch, grid_radius);
        patches.push_back(std::move(patch));
    };

    struct DigCandidate {
        Vec3 center;
        float radius_mesh = 0.0f;
        float distance_to_camera = 0.0f;
        uint32_t newest_rank = 0;
        uint32_t depth = MaxSvoDepth;
    };

    const Vec3 camera_center = length(config.lod_camera_position) > 0.000001f
        ? normalize(config.lod_camera_position)
        : Vec3{0.0f, 0.0f, 1.0f};
    if (config.voxel_edits.digs.empty() && config.voxel_edits.terrain_masks.empty() && fallback_depth >= MaxSvoDepth) {
        const float startup_radius = kilometers_to_world_units(config.local_surface_net_patch_radius_km);
        add_patch(camera_center, startup_radius, fallback_depth, true);
    }

    std::vector<DigCandidate> dig_candidates;
    dig_candidates.reserve(config.voxel_edits.digs.size() + config.voxel_edits.terrain_masks.size());
    for (uint32_t i = 0; i < config.voxel_edits.digs.size(); ++i) {
        const VoxelDigEdit& dig = config.voxel_edits.digs[i];
        if (!dig_target_matches(dig, VoxelDigTarget::Terrain) || length(dig.center_mesh) <= 0.000001f) {
            continue;
        }
        const float radius_mesh = kilometers_to_world_units(dig.radius_km);
        if (radius_mesh <= 0.0f) {
            continue;
        }
        dig_candidates.push_back({
            dig.center_mesh,
            radius_mesh,
            length(normalize(dig.center_mesh) - camera_center),
            static_cast<uint32_t>(config.voxel_edits.digs.size() - i),
            std::clamp(dig.depth > 0u ? dig.depth : fallback_depth, 1u, MaxSvoDepth),
        });
    }
    for (uint32_t i = 0; i < config.voxel_edits.terrain_masks.size(); ++i) {
        Vec3 center{};
        float radius_mesh = 0.0f;
        if (!terrain_height_mask_hole_footprint(config.voxel_edits.terrain_masks[i], center, radius_mesh)) {
            continue;
        }
        dig_candidates.push_back({
            center,
            radius_mesh,
            length(normalize(center) - camera_center),
            static_cast<uint32_t>(config.voxel_edits.digs.size() + config.voxel_edits.terrain_masks.size() - i),
            fallback_depth,
        });
    }
    std::sort(
        dig_candidates.begin(),
        dig_candidates.end(),
        [](const DigCandidate& lhs, const DigCandidate& rhs) {
            if (std::fabs(lhs.distance_to_camera - rhs.distance_to_camera) > 0.000001f) {
                return lhs.distance_to_camera < rhs.distance_to_camera;
            }
            return lhs.newest_rank < rhs.newest_rank;
        }
    );
    for (const DigCandidate& dig : dig_candidates) {
        add_patch(dig.center, dig.radius_mesh, dig.depth, true);
    }

    return patches;
}

bool surface_net_key_in_patch(Vec3 center, float leaf_radius, const std::vector<LocalSurfaceNetPatch>& patches) {
    for (const LocalSurfaceNetPatch& patch : patches) {
        if (!patch.replace_surface) {
            continue;
        }
        if (length(center - patch.center_mesh) <= patch.suppress_radius_mesh + leaf_radius) {
            return true;
        }
    }
    return false;
}

int32_t surface_net_cube_vertex(
    const std::vector<int32_t>& cube_vertices,
    uint32_t cube_resolution,
    uint32_t x,
    uint32_t y,
    uint32_t z
) {
    if (x >= cube_resolution || y >= cube_resolution || z >= cube_resolution) {
        return -1;
    }
    return cube_vertices[surface_net_cube_index(x, y, z, cube_resolution)];
}

int32_t surface_net_sparse_cube_vertex(
    const std::vector<VoxelKey>& cube_keys,
    const std::vector<int32_t>& cube_vertices,
    VoxelKey key
) {
    const auto found = std::lower_bound(cube_keys.begin(), cube_keys.end(), key, voxel_key_morton_less);
    if (found == cube_keys.end() || !(*found == key)) {
        return -1;
    }
    return cube_vertices[static_cast<size_t>(found - cube_keys.begin())];
}

void append_surface_net_quad(
    SurfaceNetMesh& surface_net,
    int32_t a,
    int32_t b,
    int32_t c,
    int32_t d
) {
    if (a < 0 || b < 0 || c < 0 || d < 0) {
        return;
    }
    surface_net.triangle_indices.push_back(static_cast<uint32_t>(a));
    surface_net.triangle_indices.push_back(static_cast<uint32_t>(b));
    surface_net.triangle_indices.push_back(static_cast<uint32_t>(c));
    surface_net.triangle_indices.push_back(static_cast<uint32_t>(a));
    surface_net.triangle_indices.push_back(static_cast<uint32_t>(c));
    surface_net.triangle_indices.push_back(static_cast<uint32_t>(d));
}

bool solve_surface_net_qef(
    const std::array<float, 6>& ata,
    const std::array<float, 3>& atb,
    Vec3& result
) {
    float m[3][4] = {
        {ata[0], ata[1], ata[2], atb[0]},
        {ata[1], ata[3], ata[4], atb[1]},
        {ata[2], ata[4], ata[5], atb[2]},
    };

    for (uint32_t column = 0; column < 3u; ++column) {
        uint32_t pivot = column;
        float pivot_value = std::fabs(m[column][column]);
        for (uint32_t row = column + 1u; row < 3u; ++row) {
            const float value = std::fabs(m[row][column]);
            if (value > pivot_value) {
                pivot = row;
                pivot_value = value;
            }
        }
        if (pivot_value <= 0.00000001f) {
            return false;
        }
        if (pivot != column) {
            for (uint32_t k = column; k < 4u; ++k) {
                std::swap(m[column][k], m[pivot][k]);
            }
        }

        const float inv_pivot = 1.0f / m[column][column];
        for (uint32_t k = column; k < 4u; ++k) {
            m[column][k] *= inv_pivot;
        }
        for (uint32_t row = 0; row < 3u; ++row) {
            if (row == column) {
                continue;
            }
            const float factor = m[row][column];
            for (uint32_t k = column; k < 4u; ++k) {
                m[row][k] -= factor * m[column][k];
            }
        }
    }

    result = {m[0][3], m[1][3], m[2][3]};
    return std::isfinite(result.x) && std::isfinite(result.y) && std::isfinite(result.z);
}

uint32_t surface_net_dense_corner_mask(
    const std::vector<uint64_t>& occupancy,
    uint32_t resolution,
    uint32_t x,
    uint32_t y,
    uint32_t z
) {
    const uint64_t row = resolution;
    const uint64_t slice = row * row;
    const uint64_t base = (static_cast<uint64_t>(z) * row + y) * row + x;
    return surface_net_bit_pair(occupancy, base) |
           (surface_net_bit_pair(occupancy, base + row) << 2u) |
           (surface_net_bit_pair(occupancy, base + slice) << 4u) |
           (surface_net_bit_pair(occupancy, base + slice + row) << 6u);
}

uint32_t surface_net_sparse_corner_mask(const std::vector<VoxelKey>& occupied_keys, VoxelKey cube) {
    uint32_t mask = 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x, cube.y, cube.z}) ? 1u << 0u : 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x + 1u, cube.y, cube.z}) ? 1u << 1u : 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x, cube.y + 1u, cube.z}) ? 1u << 2u : 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x + 1u, cube.y + 1u, cube.z}) ? 1u << 3u : 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x, cube.y, cube.z + 1u}) ? 1u << 4u : 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x + 1u, cube.y, cube.z + 1u}) ? 1u << 5u : 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x, cube.y + 1u, cube.z + 1u}) ? 1u << 6u : 0u;
    mask |= contains_voxel_key(occupied_keys, {cube.x + 1u, cube.y + 1u, cube.z + 1u}) ? 1u << 7u : 0u;
    return mask;
}

struct SurfaceNetMaskPattern {
    std::array<std::array<uint8_t, 2>, 12> edges = {};
    uint8_t edge_count = 0;
};

const SurfaceNetMaskPattern& surface_net_mask_pattern(uint32_t inside_mask) {
    static const std::array<SurfaceNetMaskPattern, 256> patterns = [] {
        static constexpr std::array<std::array<uint8_t, 2>, 12> edge_corners = {{
            {{0, 1}}, {{0, 2}}, {{1, 3}}, {{2, 3}},
            {{4, 5}}, {{4, 6}}, {{5, 7}}, {{6, 7}},
            {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
        }};

        std::array<SurfaceNetMaskPattern, 256> result = {};
        for (uint32_t mask = 0; mask < result.size(); ++mask) {
            SurfaceNetMaskPattern pattern;
            for (const auto& edge : edge_corners) {
                const bool inside_a = (mask & (1u << edge[0])) != 0u;
                const bool inside_b = (mask & (1u << edge[1])) != 0u;
                if (inside_a == inside_b) {
                    continue;
                }
                pattern.edges[pattern.edge_count++] = edge;
            }
            result[mask] = pattern;
        }
        return result;
    }();
    return patterns[inside_mask & 0xffu];
}

SurfaceNetPlacement surface_net_mask_placement(
    uint32_t inside_mask,
    Vec3 base,
    float voxel_size,
    bool enable_dual_contouring
) {
    static constexpr std::array<std::array<uint8_t, 3>, 8> corner_offsets = {{
        {{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{1, 1, 0}},
        {{0, 0, 1}}, {{1, 0, 1}}, {{0, 1, 1}}, {{1, 1, 1}},
    }};

    Vec3 average = {};
    Vec3 normal_sum = {};
    uint32_t crossing_count = 0u;
    std::array<float, 6> ata = {};
    std::array<float, 3> atb = {};
    const SurfaceNetMaskPattern& pattern = surface_net_mask_pattern(inside_mask);

    for (uint32_t edge_index = 0; edge_index < pattern.edge_count; ++edge_index) {
        const auto& edge = pattern.edges[edge_index];
        const uint32_t a = edge[0];
        const uint32_t b = edge[1];
        const bool inside_a = (inside_mask & (1u << a)) != 0u;

        const float ax = static_cast<float>(corner_offsets[a][0]);
        const float ay = static_cast<float>(corner_offsets[a][1]);
        const float az = static_cast<float>(corner_offsets[a][2]);
        const float bx = static_cast<float>(corner_offsets[b][0]);
        const float by = static_cast<float>(corner_offsets[b][1]);
        const float bz = static_cast<float>(corner_offsets[b][2]);
        const Vec3 midpoint = {
            base.x + (ax + bx) * 0.5f * voxel_size,
            base.y + (ay + by) * 0.5f * voxel_size,
            base.z + (az + bz) * 0.5f * voxel_size,
        };

        float nx = (bx - ax) * (inside_a ? 1.0f : -1.0f);
        float ny = (by - ay) * (inside_a ? 1.0f : -1.0f);
        float nz = (bz - az) * (inside_a ? 1.0f : -1.0f);
        const float midpoint_len_sq = midpoint.x * midpoint.x + midpoint.y * midpoint.y + midpoint.z * midpoint.z;
        if (midpoint_len_sq > 0.000000000001f) {
            const float inv_midpoint_len = fast_rsqrt_positive(midpoint_len_sq);
            const float rx = midpoint.x * inv_midpoint_len;
            const float ry = midpoint.y * inv_midpoint_len;
            const float rz = midpoint.z * inv_midpoint_len;
            if (nx * rx + ny * ry + nz * rz < 0.0f) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            const Vec3 biased = {nx * 0.75f + rx * 0.25f, ny * 0.75f + ry * 0.25f, nz * 0.75f + rz * 0.25f};
            const Vec3 normal = fast_normalize_vec3(biased);
            nx = normal.x;
            ny = normal.y;
            nz = normal.z;
        }

        average.x += midpoint.x;
        average.y += midpoint.y;
        average.z += midpoint.z;
        normal_sum.x += nx;
        normal_sum.y += ny;
        normal_sum.z += nz;
        ++crossing_count;

        const float plane_offset = nx * midpoint.x + ny * midpoint.y + nz * midpoint.z;
        ata[0] += nx * nx;
        ata[1] += nx * ny;
        ata[2] += nx * nz;
        ata[3] += ny * ny;
        ata[4] += ny * nz;
        ata[5] += nz * nz;
        atb[0] += nx * plane_offset;
        atb[1] += ny * plane_offset;
        atb[2] += nz * plane_offset;
    }

    if (crossing_count == 0u) {
        return {};
    }

    const float inv_crossings = 1.0f / static_cast<float>(crossing_count);
    average.x *= inv_crossings;
    average.y *= inv_crossings;
    average.z *= inv_crossings;

    Vec3 position = average;
    if (enable_dual_contouring) {
        constexpr float AnchorWeight = 0.05f;
        ata[0] += AnchorWeight;
        ata[3] += AnchorWeight;
        ata[5] += AnchorWeight;
        atb[0] += average.x * AnchorWeight;
        atb[1] += average.y * AnchorWeight;
        atb[2] += average.z * AnchorWeight;

        Vec3 solved = {};
        if (solve_surface_net_qef(ata, atb, solved)) {
            const Vec3 maximum = {base.x + voxel_size, base.y + voxel_size, base.z + voxel_size};
            position = clamp_vec3(solved, base, maximum);
        }
    }

    const Vec3 fallback_normal = fast_normalize_vec3(average);
    const float normal_sum_len_sq = normal_sum.x * normal_sum.x + normal_sum.y * normal_sum.y + normal_sum.z * normal_sum.z;
    return {
        position,
        normal_sum_len_sq > 0.000000000001f ? fast_normalize_vec3(normal_sum) : fallback_normal,
    };
}

SurfaceNetMesh build_sparse_surface_net_mesh_from_occupancy(
    std::vector<VoxelKey> occupied_keys,
    uint32_t source_depth,
    uint32_t target_depth,
    float grid_radius,
    const MarchingCubesConfig& config,
    double progress_begin,
    double progress_end
) {
    SurfaceNetMesh surface_net;
    if (!config.enable_surface_net_generation || occupied_keys.empty()) {
        return surface_net;
    }

    const uint32_t resolution = 1u << target_depth;
    const uint32_t cube_resolution = resolution - 1u;
    const float voxel_size = (grid_radius * 2.0f) / static_cast<float>(resolution);
    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    if (source_depth != target_depth) {
        occupied_keys = coarsen_surface_net_keys(occupied_keys, source_depth, target_depth);
    }

    std::vector<VoxelKey> cube_candidates;
    cube_candidates.reserve(std::min<size_t>(occupied_keys.size() * 8u, static_cast<size_t>(max_vertices) * 2u));
    std::vector<SurfaceNetEdgeKey> edge_candidates;
    edge_candidates.reserve(occupied_keys.size() * 6u);
    const double candidate_progress_end = progress_begin + (progress_end - progress_begin) * 0.30;
    const double sort_progress_end = progress_begin + (progress_end - progress_begin) * 0.38;
    const double vertex_progress_end = progress_begin + (progress_end - progress_begin) * 0.78;
    const uint64_t occupied_total = occupied_keys.size();
    uint64_t occupied_processed = 0;
    report_progress(config, progress_begin, "Collecting surface-net candidates");
    for (VoxelKey key : occupied_keys) {
        ++occupied_processed;
        report_index_progress(config, progress_begin, candidate_progress_end, occupied_processed, occupied_total, nullptr);
        if (key.x >= resolution || key.y >= resolution || key.z >= resolution) {
            continue;
        }
        for (uint32_t dz = 0; dz < 2; ++dz) {
            for (uint32_t dy = 0; dy < 2; ++dy) {
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    if ((dx == 1u && key.x == 0u) || (dy == 1u && key.y == 0u) || (dz == 1u && key.z == 0u)) {
                        continue;
                    }
                    const VoxelKey cube = {key.x - dx, key.y - dy, key.z - dz};
                    if (cube.x < cube_resolution && cube.y < cube_resolution && cube.z < cube_resolution) {
                        cube_candidates.push_back(cube);
                    }
                }
            }
        }

        auto neighbor_occupied = [&](uint32_t x, uint32_t y, uint32_t z) {
            return contains_voxel_key(occupied_keys, {x, y, z});
        };
        if (key.x > 0u && !neighbor_occupied(key.x - 1u, key.y, key.z)) edge_candidates.push_back({key.x - 1u, key.y, key.z, 0u});
        if (key.x + 1u < resolution && !neighbor_occupied(key.x + 1u, key.y, key.z)) edge_candidates.push_back({key.x, key.y, key.z, 0u});
        if (key.y > 0u && !neighbor_occupied(key.x, key.y - 1u, key.z)) edge_candidates.push_back({key.x, key.y - 1u, key.z, 1u});
        if (key.y + 1u < resolution && !neighbor_occupied(key.x, key.y + 1u, key.z)) edge_candidates.push_back({key.x, key.y, key.z, 1u});
        if (key.z > 0u && !neighbor_occupied(key.x, key.y, key.z - 1u)) edge_candidates.push_back({key.x, key.y, key.z - 1u, 2u});
        if (key.z + 1u < resolution && !neighbor_occupied(key.x, key.y, key.z + 1u)) edge_candidates.push_back({key.x, key.y, key.z, 2u});
    }

    report_progress(config, candidate_progress_end, "Morton-sorting surface-net candidates");
    sort_and_unique_voxel_keys(cube_candidates);
    sort_and_unique_surface_net_edge_keys(edge_candidates);
    report_progress(config, sort_progress_end, "Placing surface-net vertices");

    std::vector<int32_t> cube_vertices(cube_candidates.size(), -1);
    surface_net.source_depth = target_depth;
    surface_net.bounds_radius = grid_radius;
    surface_net.occupied_voxel_count = static_cast<uint32_t>(occupied_keys.size());
    surface_net.candidate_cube_count = static_cast<uint32_t>(cube_candidates.size());
    surface_net.material_id = config.surface_net_material_id;
    surface_net.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    surface_net.local_edit_depth = config.voxel_edits.digs.empty() ? 0u : config.voxel_edits.local_depth;
    surface_net.vertices.reserve(std::min<uint32_t>(surface_net.candidate_cube_count, max_vertices));
    surface_net.normals.reserve(surface_net.vertices.capacity());

    const uint64_t cube_total = cube_candidates.size();
    for (size_t cube_index = 0; cube_index < cube_candidates.size(); ++cube_index) {
        report_index_progress(config, sort_progress_end, vertex_progress_end, cube_index + 1u, cube_total, nullptr);
        if (surface_net.vertices.size() >= max_vertices) {
            break;
        }
        const VoxelKey cube = cube_candidates[cube_index];

        const uint32_t inside_mask = surface_net_sparse_corner_mask(occupied_keys, cube);
        const uint32_t inside_count = std::popcount(inside_mask);
        if (inside_count == 0u || inside_count == 8u) {
            continue;
        }

        const SurfaceNetPlacement placement = surface_net_mask_placement(
            inside_mask,
            surface_net_sample_position(cube.x, cube.y, cube.z, grid_radius, voxel_size),
            voxel_size,
            config.enable_surface_net_dual_contouring
        );
        if (length(placement.normal) <= 0.000001f) {
            continue;
        }

        const uint32_t vertex_index = static_cast<uint32_t>(surface_net.vertices.size());
        surface_net.vertices.push_back(placement.position);
        surface_net.normals.push_back(placement.normal);
        surface_net.vertex_depths.push_back(target_depth);
        cube_vertices[cube_index] = static_cast<int32_t>(vertex_index);
    }

    report_progress(config, vertex_progress_end, "Connecting surface-net quads");
    surface_net.triangle_indices.reserve(edge_candidates.size() * 6u);
    const uint64_t edge_total = edge_candidates.size();
    uint64_t edge_processed = 0;
    for (SurfaceNetEdgeKey edge : edge_candidates) {
        ++edge_processed;
        report_index_progress(config, vertex_progress_end, progress_end, edge_processed, edge_total, nullptr);
        if (edge.axis == 0u) {
            if (edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y - 1u, edge.z - 1u}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y, edge.z - 1u}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y, edge.z}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y - 1u, edge.z})
            );
        } else if (edge.axis == 1u) {
            if (edge.y >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x - 1u, edge.y, edge.z - 1u}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y, edge.z - 1u}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y, edge.z}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x - 1u, edge.y, edge.z})
            );
        } else {
            if (edge.z >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x - 1u, edge.y - 1u, edge.z}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y - 1u, edge.z}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x, edge.y, edge.z}),
                surface_net_sparse_cube_vertex(cube_candidates, cube_vertices, {edge.x - 1u, edge.y, edge.z})
            );
        }
    }

    return surface_net;
}

SurfaceNetMesh build_surface_net_mesh_from_occupancy(
    const std::vector<VoxelKey>& sorted_occupied_keys,
    uint32_t source_depth,
    float grid_radius,
    const MarchingCubesConfig& config,
    const std::vector<LocalSurfaceNetPatch>& suppression_patches = {},
    double progress_begin = 0.72,
    double progress_end = 0.80
) {
    SurfaceNetMesh surface_net;
    if (!config.enable_surface_net_generation || sorted_occupied_keys.empty()) {
        return surface_net;
    }

    const uint32_t target_depth = std::clamp(std::min(config.surface_net_depth, source_depth), 1u, MaxSvoDepth);
    if (target_depth > 8u) {
        return build_sparse_surface_net_mesh_from_occupancy(
            sorted_occupied_keys,
            source_depth,
            target_depth,
            grid_radius,
            config,
            progress_begin,
            progress_end
        );
    }
    const uint32_t resolution = 1u << target_depth;
    const uint32_t cube_resolution = resolution - 1u;
    const float voxel_size = (grid_radius * 2.0f) / static_cast<float>(resolution);
    const float voxel_radius = std::sqrt(3.0f) * voxel_size * 0.5f;
    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    std::vector<VoxelKey> occupied_keys = coarsen_surface_net_keys(sorted_occupied_keys, source_depth, target_depth);
    if (!suppression_patches.empty()) {
        occupied_keys.erase(
            std::remove_if(
                occupied_keys.begin(),
                occupied_keys.end(),
                [&](VoxelKey key) {
                    (void)voxel_radius;
                    return target_depth == SurfacePromotionRootDepth &&
                           depth8_key_in_promoted_patches(key, suppression_patches);
                }
            ),
            occupied_keys.end()
        );
    }
    const uint32_t grid_count = resolution * resolution * resolution;
    std::vector<uint64_t> occupancy((grid_count + 63u) / 64u, 0u);
    for (VoxelKey key : occupied_keys) {
        if (key.x < resolution && key.y < resolution && key.z < resolution) {
            set_surface_net_bit(occupancy, surface_net_grid_index(key.x, key.y, key.z, resolution));
        }
    }

    const uint32_t cube_count = cube_resolution * cube_resolution * cube_resolution;
    std::vector<uint64_t> cube_candidate_bits((cube_count + 63u) / 64u, 0u);
    const uint64_t edge_candidate_count = static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * 3ull;
    std::vector<uint64_t> edge_candidate_bits(static_cast<size_t>((edge_candidate_count + 63ull) / 64ull), 0u);
    const double candidate_progress_end = progress_begin + (progress_end - progress_begin) * 0.30;
    const double sort_progress_end = progress_begin + (progress_end - progress_begin) * 0.38;
    const double vertex_progress_end = progress_begin + (progress_end - progress_begin) * 0.78;
    const uint64_t occupied_total = occupied_keys.size();
    uint64_t occupied_processed = 0;
    report_progress(config, progress_begin, "Collecting surface-net candidates");
    for (VoxelKey key : occupied_keys) {
        ++occupied_processed;
        report_index_progress(config, progress_begin, candidate_progress_end, occupied_processed, occupied_total, nullptr);
        if (key.x >= resolution || key.y >= resolution || key.z >= resolution) {
            continue;
        }
        for (uint32_t dz = 0; dz < 2; ++dz) {
            for (uint32_t dy = 0; dy < 2; ++dy) {
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    if ((dx == 1u && key.x == 0u) || (dy == 1u && key.y == 0u) || (dz == 1u && key.z == 0u)) {
                        continue;
                    }
                    const uint32_t cube_x = key.x - dx;
                    const uint32_t cube_y = key.y - dy;
                    const uint32_t cube_z = key.z - dz;
                    set_surface_net_cube_candidate(cube_candidate_bits, cube_resolution, cube_x, cube_y, cube_z);
                }
            }
        }

        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 0u, false);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 0u, true);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 1u, false);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 1u, true);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 2u, false);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 2u, true);
    }

    report_progress(config, candidate_progress_end, "Compacting surface-net candidates");
    const double cube_compact_progress_end = candidate_progress_end + (sort_progress_end - candidate_progress_end) * 0.40;
    std::vector<VoxelKey> cube_candidates = compact_surface_net_cube_candidates(
        cube_candidate_bits,
        cube_resolution,
        config,
        candidate_progress_end,
        cube_compact_progress_end
    );
    std::vector<SurfaceNetEdgeKey> edge_candidates = compact_surface_net_edge_candidates(
        edge_candidate_bits,
        resolution,
        config,
        cube_compact_progress_end,
        sort_progress_end
    );
    report_progress(config, sort_progress_end, "Placing surface-net vertices");

    std::vector<int32_t> cube_vertices(cube_count, -1);
    surface_net.source_depth = target_depth;
    surface_net.bounds_radius = grid_radius;
    surface_net.occupied_voxel_count = static_cast<uint32_t>(occupied_keys.size());
    surface_net.candidate_cube_count = static_cast<uint32_t>(cube_candidates.size());
    surface_net.material_id = config.surface_net_material_id;
    surface_net.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    surface_net.local_edit_depth = config.voxel_edits.digs.empty() ? 0u : config.voxel_edits.local_depth;
    surface_net.vertices.reserve(std::min<uint32_t>(surface_net.candidate_cube_count, max_vertices));
    surface_net.normals.reserve(surface_net.vertices.capacity());

    const uint64_t cube_total = cube_candidates.size();
    uint64_t cube_processed = 0;
    for (VoxelKey cube : cube_candidates) {
        ++cube_processed;
        report_index_progress(config, sort_progress_end, vertex_progress_end, cube_processed, cube_total, nullptr);
        if (surface_net.vertices.size() >= max_vertices) {
            break;
        }

        const uint32_t inside_mask = surface_net_dense_corner_mask(occupancy, resolution, cube.x, cube.y, cube.z);
        const uint32_t inside_count = std::popcount(inside_mask);
        if (inside_count == 0u || inside_count == 8u) {
            continue;
        }

        const SurfaceNetPlacement placement = surface_net_mask_placement(
            inside_mask,
            surface_net_sample_position(cube.x, cube.y, cube.z, grid_radius, voxel_size),
            voxel_size,
            config.enable_surface_net_dual_contouring
        );
        if (length(placement.normal) <= 0.000001f) {
            continue;
        }

        const uint32_t vertex_index = static_cast<uint32_t>(surface_net.vertices.size());
        surface_net.vertices.push_back(placement.position);
        surface_net.normals.push_back(placement.normal);
        surface_net.vertex_depths.push_back(target_depth);
        cube_vertices[surface_net_cube_index(cube.x, cube.y, cube.z, cube_resolution)] = static_cast<int32_t>(vertex_index);
    }

    report_progress(config, vertex_progress_end, "Connecting surface-net quads");
    surface_net.triangle_indices.reserve(edge_candidates.size() * 6u);
    const uint64_t edge_total = edge_candidates.size();
    uint64_t edge_processed = 0;
    for (SurfaceNetEdgeKey edge : edge_candidates) {
        ++edge_processed;
        report_index_progress(config, vertex_progress_end, progress_end, edge_processed, edge_total, nullptr);
        if (edge.axis == 0u) {
            if (edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y - 1u, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y - 1u, edge.z)
            );
        } else if (edge.axis == 1u) {
            if (edge.y >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y, edge.z)
            );
        } else {
            if (edge.z >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y - 1u, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y - 1u, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y, edge.z)
            );
        }
    }

    return surface_net;
}

void append_surface_net_mesh(SurfaceNetMesh& destination, const SurfaceNetMesh& source) {
    const uint32_t vertex_offset = static_cast<uint32_t>(destination.vertices.size());
    destination.vertices.insert(destination.vertices.end(), source.vertices.begin(), source.vertices.end());
    destination.normals.insert(destination.normals.end(), source.normals.begin(), source.normals.end());
    destination.vertex_depths.reserve(destination.vertex_depths.size() + source.vertices.size());
    if (source.vertex_depths.size() == source.vertices.size()) {
        destination.vertex_depths.insert(destination.vertex_depths.end(), source.vertex_depths.begin(), source.vertex_depths.end());
    } else {
        destination.vertex_depths.insert(destination.vertex_depths.end(), source.vertices.size(), source.source_depth);
    }
    destination.triangle_indices.reserve(destination.triangle_indices.size() + source.triangle_indices.size());
    for (uint32_t index : source.triangle_indices) {
        destination.triangle_indices.push_back(vertex_offset + index);
    }
}

Vec3 cave_local_to_world(const LocalVoxelFeature& feature, Vec3 local) {
    return feature.center_mesh +
           feature.tangent_mesh * local.x +
           feature.bitangent_mesh * local.y -
           feature.normal_mesh * local.z;
}

Vec3 cave_world_to_local(const LocalVoxelFeature& feature, Vec3 world) {
    const Vec3 offset = world - feature.center_mesh;
    return {
        dot(offset, feature.tangent_mesh),
        dot(offset, feature.bitangent_mesh),
        -dot(offset, feature.normal_mesh),
    };
}

float cave_tunnel_sdf(Vec3 local, float entrance_radius, float tunnel_radius, float depth) {
    const float z0 = -entrance_radius * 0.45f;
    const float z1 = depth;
    const float t = std::clamp((local.z - z0) / std::max(0.000001f, z1 - z0), 0.0f, 1.0f);
    const float radius = entrance_radius * (1.0f - t) + tunnel_radius * t;
    const float radial = std::sqrt(local.x * local.x + local.y * local.y) - radius;
    const float lower_cap = z0 - local.z;
    const float upper_cap = local.z - z1;
    return std::max(radial, std::max(lower_cap, upper_cap));
}

float cave_air_sdf(const LocalVoxelFeature& feature, Vec3 local, const VoxelEditSet& edits) {
    const float entrance_radius = kilometers_to_world_units(feature.entrance_radius_km);
    const float tunnel_radius = kilometers_to_world_units(feature.tunnel_radius_km);
    const float chamber_radius = kilometers_to_world_units(feature.chamber_radius_km);
    const float depth = kilometers_to_world_units(feature.depth_km);

    float sdf = cave_tunnel_sdf(local, entrance_radius, tunnel_radius, depth);

    const uint32_t side_hash = hash_u32(feature.seed ^ 0x9e3779b9u);
    const float angle = (static_cast<float>(side_hash & 0xffffu) / 65535.0f) * 2.0f * Pi;
    const float side_distance = chamber_radius * 0.42f;
    const Vec3 chamber_center{
        std::cos(angle) * side_distance,
        std::sin(angle) * side_distance,
        depth * 0.86f,
    };
    sdf = std::min(sdf, length(local - chamber_center) - chamber_radius);

    const uint32_t lobe_hash = hash_u32(feature.seed ^ 0x51ed270bu);
    const float lobe_angle = (static_cast<float>(lobe_hash & 0xffffu) / 65535.0f) * 2.0f * Pi;
    const Vec3 lobe_center{
        std::cos(lobe_angle) * chamber_radius * 0.72f,
        std::sin(lobe_angle) * chamber_radius * 0.72f,
        depth * 0.58f,
    };
    sdf = std::min(sdf, length(local - lobe_center) - chamber_radius * 0.48f);

    for (const VoxelDigEdit& dig : edits.digs) {
        if (!dig_target_matches(dig, VoxelDigTarget::CaveInterior)) {
            continue;
        }
        const Vec3 edit_local = cave_world_to_local(feature, dig.center_mesh);
        const float edit_radius = kilometers_to_world_units(dig.radius_km);
        if (std::fabs(edit_local.z - local.z) > edit_radius + chamber_radius) {
            continue;
        }
        sdf = std::min(sdf, length(local - edit_local) - edit_radius);
    }

    return sdf;
}

struct CaveAirSphere {
    Vec3 center;
    float radius = 0.0f;
};

struct CaveAirField {
    float entrance_radius = 0.0f;
    float tunnel_radius = 0.0f;
    float chamber_radius = 0.0f;
    float depth = 0.0f;
    float tunnel_z0 = 0.0f;
    float tunnel_z1 = 0.0f;
    CaveAirSphere chamber;
    CaveAirSphere lobe;
    std::vector<CaveAirSphere> edits;
};

CaveAirField build_cave_air_field(const LocalVoxelFeature& feature, const VoxelEditSet& edits) {
    CaveAirField field;
    field.entrance_radius = kilometers_to_world_units(feature.entrance_radius_km);
    field.tunnel_radius = kilometers_to_world_units(feature.tunnel_radius_km);
    field.chamber_radius = kilometers_to_world_units(feature.chamber_radius_km);
    field.depth = kilometers_to_world_units(feature.depth_km);
    field.tunnel_z0 = -field.entrance_radius * 0.45f;
    field.tunnel_z1 = field.depth;

    const uint32_t side_hash = hash_u32(feature.seed ^ 0x9e3779b9u);
    const float angle = (static_cast<float>(side_hash & 0xffffu) / 65535.0f) * 2.0f * Pi;
    const float side_distance = field.chamber_radius * 0.42f;
    field.chamber = {
        {
            std::cos(angle) * side_distance,
            std::sin(angle) * side_distance,
            field.depth * 0.86f,
        },
        field.chamber_radius,
    };

    const uint32_t lobe_hash = hash_u32(feature.seed ^ 0x51ed270bu);
    const float lobe_angle = (static_cast<float>(lobe_hash & 0xffffu) / 65535.0f) * 2.0f * Pi;
    field.lobe = {
        {
            std::cos(lobe_angle) * field.chamber_radius * 0.72f,
            std::sin(lobe_angle) * field.chamber_radius * 0.72f,
            field.depth * 0.58f,
        },
        field.chamber_radius * 0.48f,
    };

    field.edits.reserve(edits.digs.size());
    for (const VoxelDigEdit& dig : edits.digs) {
        if (!dig_target_matches(dig, VoxelDigTarget::CaveInterior)) {
            continue;
        }
        field.edits.push_back({
            cave_world_to_local(feature, dig.center_mesh),
            kilometers_to_world_units(dig.radius_km),
        });
    }
    return field;
}

float cave_air_sdf(const CaveAirField& field, Vec3 local) {
    float sdf = cave_tunnel_sdf(local, field.entrance_radius, field.tunnel_radius, field.depth);
    sdf = std::min(sdf, length(local - field.chamber.center) - field.chamber.radius);
    sdf = std::min(sdf, length(local - field.lobe.center) - field.lobe.radius);

    for (const CaveAirSphere& edit : field.edits) {
        if (std::fabs(edit.center.z - local.z) > edit.radius + field.chamber_radius) {
            continue;
        }
        sdf = std::min(sdf, length(local - edit.center) - edit.radius);
    }

    return sdf;
}

bool cave_air_contains(const CaveAirField& field, float x, float y, float z) {
    if (z >= field.tunnel_z0 && z <= field.tunnel_z1) {
        const float t = std::clamp(
            (z - field.tunnel_z0) / std::max(0.000001f, field.tunnel_z1 - field.tunnel_z0),
            0.0f,
            1.0f
        );
        const float radius = field.entrance_radius * (1.0f - t) + field.tunnel_radius * t;
        if ((x * x + y * y) <= radius * radius) {
            return true;
        }
    }

    auto inside_sphere = [&](const CaveAirSphere& sphere) {
        const float dx = x - sphere.center.x;
        const float dy = y - sphere.center.y;
        const float dz = z - sphere.center.z;
        return (dx * dx + dy * dy + dz * dz) <= sphere.radius * sphere.radius;
    };

    if (inside_sphere(field.chamber) || inside_sphere(field.lobe)) {
        return true;
    }
    for (const CaveAirSphere& edit : field.edits) {
        if (inside_sphere(edit)) {
            return true;
        }
    }
    return false;
}

bool cave_air_contains(const CaveAirField& field, Vec3 local) {
    return cave_air_contains(field, local.x, local.y, local.z);
}

struct CaveSliceBounds {
    uint32_t min_x = 0;
    uint32_t max_x = 0;
    uint32_t min_y = 0;
    uint32_t max_y = 0;
    bool any = false;
};

void include_cave_slice_aabb(
    CaveSliceBounds& bounds,
    Vec3 origin,
    float step,
    uint32_t resolution,
    float min_x,
    float max_x,
    float min_y,
    float max_y
) {
    const float pad = step * 2.0f;
    const int32_t x0 = static_cast<int32_t>(std::floor((min_x - pad - origin.x) / step));
    const int32_t x1 = static_cast<int32_t>(std::ceil((max_x + pad - origin.x) / step));
    const int32_t y0 = static_cast<int32_t>(std::floor((min_y - pad - origin.y) / step));
    const int32_t y1 = static_cast<int32_t>(std::ceil((max_y + pad - origin.y) / step));
    const int32_t max_index = static_cast<int32_t>(resolution) - 1;
    if (x1 < 0 || y1 < 0 || x0 > max_index || y0 > max_index) {
        return;
    }

    const uint32_t clamped_x0 = static_cast<uint32_t>(std::clamp(x0, 0, max_index));
    const uint32_t clamped_x1 = static_cast<uint32_t>(std::clamp(x1, 0, max_index));
    const uint32_t clamped_y0 = static_cast<uint32_t>(std::clamp(y0, 0, max_index));
    const uint32_t clamped_y1 = static_cast<uint32_t>(std::clamp(y1, 0, max_index));
    if (!bounds.any) {
        bounds.min_x = clamped_x0;
        bounds.max_x = clamped_x1;
        bounds.min_y = clamped_y0;
        bounds.max_y = clamped_y1;
        bounds.any = true;
        return;
    }

    bounds.min_x = std::min(bounds.min_x, clamped_x0);
    bounds.max_x = std::max(bounds.max_x, clamped_x1);
    bounds.min_y = std::min(bounds.min_y, clamped_y0);
    bounds.max_y = std::max(bounds.max_y, clamped_y1);
}

CaveSliceBounds cave_slice_bounds(
    const CaveAirField& field,
    Vec3 origin,
    float step,
    uint32_t resolution,
    float z
) {
    CaveSliceBounds bounds;
    if (z >= field.tunnel_z0 - step && z <= field.tunnel_z1 + step) {
        const float t = std::clamp(
            (z - field.tunnel_z0) / std::max(0.000001f, field.tunnel_z1 - field.tunnel_z0),
            0.0f,
            1.0f
        );
        const float radius = field.entrance_radius * (1.0f - t) + field.tunnel_radius * t;
        include_cave_slice_aabb(bounds, origin, step, resolution, -radius, radius, -radius, radius);
    }

    auto include_sphere = [&](const CaveAirSphere& sphere) {
        const float dz = z - sphere.center.z;
        const float padded_radius = sphere.radius + step;
        if (std::fabs(dz) > padded_radius) {
            return;
        }
        const float slice_radius = std::sqrt(std::max(0.0f, padded_radius * padded_radius - dz * dz));
        include_cave_slice_aabb(
            bounds,
            origin,
            step,
            resolution,
            sphere.center.x - slice_radius,
            sphere.center.x + slice_radius,
            sphere.center.y - slice_radius,
            sphere.center.y + slice_radius
        );
    };

    include_sphere(field.chamber);
    include_sphere(field.lobe);
    for (const CaveAirSphere& edit : field.edits) {
        include_sphere(edit);
    }
    return bounds;
}

uint32_t cave_surface_net_resolution(uint32_t depth) {
    if (depth >= MaxSvoDepth) {
        return 112u;
    }
    if (depth >= 13u) {
        return 88u;
    }
    if (depth <= 4u) {
        return 24u;
    }
    if (depth <= 8u) {
        return 36u;
    }
    return 56u;
}

SurfaceNetMesh build_cave_surface_net_feature(
    const LocalVoxelFeature& feature,
    const MarchingCubesConfig& config,
    float grid_radius,
    std::vector<VoxelKey>& sign_changing_keys,
    CaveFeatureBuildStats* stats = nullptr
) {
    SurfaceNetMesh surface_net;
    if (!config.enable_surface_net_generation) {
        return surface_net;
    }

    const uint32_t depth = std::clamp(feature.svo_depth, 1u, MaxSvoDepth);
    const uint32_t resolution = cave_surface_net_resolution(depth);
    const uint32_t cube_resolution = resolution - 1u;
    const float entrance_radius = kilometers_to_world_units(feature.entrance_radius_km);
    const float tunnel_radius = kilometers_to_world_units(feature.tunnel_radius_km);
    const float chamber_radius = kilometers_to_world_units(feature.chamber_radius_km);
    const float cave_depth = kilometers_to_world_units(feature.depth_km);
    const float lateral_extent = std::max({entrance_radius * 1.75f, chamber_radius * 1.55f, tunnel_radius * 2.2f});
    const float z_min = -entrance_radius * 0.62f;
    const float z_max = cave_depth + chamber_radius * 1.45f;
    const Vec3 local_min{-lateral_extent, -lateral_extent, z_min};
    const Vec3 local_max{lateral_extent, lateral_extent, z_max};
    const Vec3 local_size = local_max - local_min;
    const float step = std::max({local_size.x, local_size.y, local_size.z}) / static_cast<float>(resolution - 1u);
    const Vec3 padded_size = Vec3{
        step * static_cast<float>(resolution - 1u),
        step * static_cast<float>(resolution - 1u),
        step * static_cast<float>(resolution - 1u),
    };
    const Vec3 origin = {
        (local_min.x + local_max.x - padded_size.x) * 0.5f,
        (local_min.y + local_max.y - padded_size.y) * 0.5f,
        (local_min.z + local_max.z - padded_size.z) * 0.5f,
    };
    CaveAirField air_field = build_cave_air_field(feature, config.voxel_edits);
    air_field.tunnel_z0 = origin.z - step * 2.0f;

    const uint32_t grid_count = resolution * resolution * resolution;
    std::vector<uint64_t> occupancy((grid_count + 63u) / 64u, 0u);
    const auto occupancy_begin = PerfClock::now();
    for (uint32_t z = 0; z < resolution; ++z) {
        const float local_z = origin.z + static_cast<float>(z) * step;
        const CaveSliceBounds bounds = cave_slice_bounds(air_field, origin, step, resolution, local_z);
        if (!bounds.any) {
            continue;
        }
        for (uint32_t y = bounds.min_y; y <= bounds.max_y; ++y) {
            for (uint32_t x = bounds.min_x; x <= bounds.max_x; ++x) {
                const Vec3 local = origin + Vec3{
                    static_cast<float>(x) * step,
                    static_cast<float>(y) * step,
                    static_cast<float>(z) * step,
                };
                if (cave_air_contains(air_field, local)) {
                    set_surface_net_bit(occupancy, surface_net_grid_index(x, y, z, resolution));
                }
            }
        }
    }
    if (stats) {
        stats->occupancy_ms = elapsed_ms(occupancy_begin);
    }

    const uint32_t cube_count = cube_resolution * cube_resolution * cube_resolution;
    std::vector<CaveSurfaceCubeCandidate> cube_candidates;
    cube_candidates.reserve(32768u);
    const uint64_t edge_candidate_count =
        static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * 3ull;
    std::vector<uint64_t> edge_candidate_bits(static_cast<size_t>((edge_candidate_count + 63ull) / 64ull), 0u);
    std::vector<SurfaceNetEdgeKey> edge_candidates;
    edge_candidates.reserve(65536u);

    const auto candidate_begin = PerfClock::now();
    for (uint32_t z = 0; z < cube_resolution; ++z) {
        const float z0 = origin.z + static_cast<float>(z) * step;
        const float z1 = z0 + step;
        const CaveSliceBounds bounds0 = cave_slice_bounds(air_field, origin, step, resolution, z0);
        const CaveSliceBounds bounds1 = cave_slice_bounds(air_field, origin, step, resolution, z1);
        if (!bounds0.any && !bounds1.any) {
            continue;
        }
        const uint32_t min_x = std::max(1u, std::min(bounds0.any ? bounds0.min_x : resolution - 1u, bounds1.any ? bounds1.min_x : resolution - 1u)) - 1u;
        const uint32_t min_y = std::max(1u, std::min(bounds0.any ? bounds0.min_y : resolution - 1u, bounds1.any ? bounds1.min_y : resolution - 1u)) - 1u;
        const uint32_t max_x = std::min(cube_resolution - 1u, std::max(bounds0.any ? bounds0.max_x : 0u, bounds1.any ? bounds1.max_x : 0u));
        const uint32_t max_y = std::min(cube_resolution - 1u, std::max(bounds0.any ? bounds0.max_y : 0u, bounds1.any ? bounds1.max_y : 0u));
        for (uint32_t y = min_y; y <= max_y; ++y) {
            for (uint32_t x = min_x; x <= max_x; ++x) {
                const uint32_t inside_mask = surface_net_dense_corner_mask(occupancy, resolution, x, y, z);
                if (inside_mask != 0u && inside_mask != 0xffu) {
                    cube_candidates.push_back({{x, y, z}, static_cast<uint8_t>(inside_mask)});
                }
            }
        }
    }

    for (uint32_t z = 0; z < resolution; ++z) {
        for (uint32_t y = 0; y < resolution; ++y) {
            for (uint32_t x = 0; x < resolution; ++x) {
                if (!surface_net_bit(occupancy, surface_net_grid_index(x, y, z, resolution))) {
                    continue;
                }
                SurfaceNetEdgeKey edge = {};
                if (set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, x, y, z, 0u, false, &edge)) {
                    edge_candidates.push_back(edge);
                }
                if (set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, x, y, z, 0u, true, &edge)) {
                    edge_candidates.push_back(edge);
                }
                if (set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, x, y, z, 1u, false, &edge)) {
                    edge_candidates.push_back(edge);
                }
                if (set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, x, y, z, 1u, true, &edge)) {
                    edge_candidates.push_back(edge);
                }
                if (set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, x, y, z, 2u, false, &edge)) {
                    edge_candidates.push_back(edge);
                }
                if (set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, x, y, z, 2u, true, &edge)) {
                    edge_candidates.push_back(edge);
                }
            }
        }
    }
    if (stats) {
        stats->candidate_ms = elapsed_ms(candidate_begin);
    }

    const auto compact_begin = PerfClock::now();
    if (stats) {
        stats->compact_ms = elapsed_ms(compact_begin);
    }

    std::vector<int32_t> cube_vertices(cube_count, -1);
    surface_net.source_depth = depth;
    surface_net.bounds_radius = grid_radius;
    surface_net.material_id = config.surface_net_material_id;
    surface_net.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    surface_net.local_edit_depth = config.voxel_edits.digs.empty() ? 0u : config.voxel_edits.local_depth;
    surface_net.occupied_voxel_count = 0u;
    surface_net.candidate_cube_count = static_cast<uint32_t>(cube_candidates.size());
    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    const size_t vertex_write_begin = surface_net.vertices.size();
    const size_t vertex_write_capacity = std::min<size_t>(cube_candidates.size(), static_cast<size_t>(max_vertices));
    surface_net.vertices.resize(vertex_write_begin + vertex_write_capacity);
    surface_net.normals.resize(vertex_write_begin + vertex_write_capacity);
    surface_net.vertex_depths.resize(vertex_write_begin + vertex_write_capacity);
    const size_t key_write_begin = sign_changing_keys.size();
    sign_changing_keys.resize(key_write_begin + vertex_write_capacity);
    size_t vertex_write_count = 0;

    const uint32_t global_resolution = 1u << depth;
    const float global_voxel_size = (grid_radius * 2.0f) / static_cast<float>(global_resolution);
    const auto vertex_begin = PerfClock::now();
    for (size_t cube_index = 0; cube_index < cube_candidates.size(); ++cube_index) {
        if (vertex_write_count >= vertex_write_capacity) {
            break;
        }
        const CaveSurfaceCubeCandidate cube_candidate = cube_candidates[cube_index];
        const VoxelKey cube = cube_candidate.key;
        const uint32_t inside_mask = cube_candidate.inside_mask;

        const Vec3 local_base = origin + Vec3{
            static_cast<float>(cube.x) * step,
            static_cast<float>(cube.y) * step,
            static_cast<float>(cube.z) * step,
        };
        const SurfaceNetPlacement placement = surface_net_mask_placement(
            inside_mask,
            local_base,
            step,
            config.enable_surface_net_dual_contouring
        );
        if (length(placement.normal) <= 0.000001f) {
            continue;
        }

        const Vec3 world_position = cave_local_to_world(feature, placement.position);
        const Vec3 world_normal = normalize(
            feature.tangent_mesh * placement.normal.x +
            feature.bitangent_mesh * placement.normal.y -
            feature.normal_mesh * placement.normal.z
        );
        const uint32_t vertex_index = static_cast<uint32_t>(vertex_write_begin + vertex_write_count);
        surface_net.vertices[vertex_index] = world_position;
        surface_net.normals[vertex_index] = world_normal;
        surface_net.vertex_depths[vertex_index] = depth;
        cube_vertices[surface_net_cube_index(cube.x, cube.y, cube.z, cube_resolution)] = static_cast<int32_t>(vertex_index);
        sign_changing_keys[key_write_begin + vertex_write_count] = voxel_key_for_position(world_position, grid_radius, global_voxel_size, global_resolution);
        ++vertex_write_count;
    }
    surface_net.vertices.resize(vertex_write_begin + vertex_write_count);
    surface_net.normals.resize(vertex_write_begin + vertex_write_count);
    surface_net.vertex_depths.resize(vertex_write_begin + vertex_write_count);
    sign_changing_keys.resize(key_write_begin + vertex_write_count);
    if (stats) {
        stats->vertex_ms = elapsed_ms(vertex_begin);
    }

    const size_t triangle_write_begin = surface_net.triangle_indices.size();
    surface_net.triangle_indices.resize(triangle_write_begin + edge_candidates.size() * 6u);
    uint32_t* triangle_write = surface_net.triangle_indices.data() + triangle_write_begin;
    const auto quad_begin = PerfClock::now();
    for (SurfaceNetEdgeKey edge : edge_candidates) {
        if (write_surface_net_edge_candidate_quad(cube_vertices, cube_resolution, edge, triangle_write)) {
            triangle_write += 6u;
        }
    }
    surface_net.triangle_indices.resize(static_cast<size_t>(triangle_write - surface_net.triangle_indices.data()));
    if (stats) {
        stats->quad_ms = elapsed_ms(quad_begin);
    }

    return surface_net;
}

void generate_cave_feature_voxels(QuantizedMesh& mesh, const MarchingCubesConfig& config) {
    const auto cave_begin = PerfClock::now();
    mesh.surface_net = {};
    mesh.svo = {};
    if (mesh.voxel_features.empty()) {
        report_progress(config, 0.97f, "No cave features");
        mesh.perf.cave_feature_count = 0;
        return;
    }

    report_progress(config, 0.56f, "Extracting cave surface nets");
    const uint32_t depth = std::clamp(config.voxel_features.cave_depth, 1u, MaxSvoDepth);
    const float grid_radius = mesh_voxel_bounds_radius(mesh);
    std::vector<VoxelKey> cave_surface_keys;
    cave_surface_keys.reserve(mesh.voxel_features.size() * 8192u);

    mesh.surface_net.source_depth = depth;
    mesh.surface_net.bounds_radius = grid_radius;
    mesh.surface_net.material_id = config.surface_net_material_id;
    mesh.surface_net.local_patch_count = static_cast<uint32_t>(mesh.voxel_features.size());
    mesh.surface_net.local_patch_depth = depth;
    mesh.surface_net.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    mesh.surface_net.local_edit_depth = config.voxel_edits.digs.empty() ? 0u : config.voxel_edits.local_depth;
    mesh.perf.cave_features.clear();
    mesh.perf.cave_features.reserve(mesh.voxel_features.size());
    mesh.perf.cave_feature_count = static_cast<uint32_t>(mesh.voxel_features.size());

    const auto cave_surface_begin = PerfClock::now();
    struct CaveFeatureExtractionResult {
        SurfaceNetMesh surface;
        std::vector<VoxelKey> sign_changing_keys;
        CaveFeatureBuildStats stats;
    };

    std::vector<std::future<CaveFeatureExtractionResult>> feature_futures;
    feature_futures.reserve(mesh.voxel_features.size());
    for (uint32_t i = 0; i < mesh.voxel_features.size(); ++i) {
        LocalVoxelFeature feature = mesh.voxel_features[i];
        feature.svo_depth = depth;
        feature_futures.push_back(std::async(
            std::launch::async,
            [feature, config, grid_radius, depth]() mutable {
                CaveFeatureExtractionResult result;
                MarchingCubesConfig worker_config = config;
                worker_config.progress_callback = {};
                result.sign_changing_keys.reserve(24576u);

                const auto feature_begin = PerfClock::now();
                result.stats.feature_id = feature.feature_id;
                result.stats.owner_cell_id = feature.owner_cell_id;
                result.stats.depth = depth;
                result.surface = build_cave_surface_net_feature(
                    feature,
                    worker_config,
                    grid_radius,
                    result.sign_changing_keys,
                    &result.stats
                );

                result.stats.vertices = static_cast<uint32_t>(result.surface.vertices.size());
                result.stats.triangles = static_cast<uint32_t>(result.surface.triangle_indices.size() / 3u);
                result.stats.occupied_voxels = static_cast<uint32_t>(result.sign_changing_keys.size());
                result.stats.candidate_cubes = result.surface.candidate_cube_count;
                result.stats.surface_net_ms = elapsed_ms(feature_begin);
                return result;
            }
        ));
    }

    std::vector<CaveFeatureExtractionResult> feature_results;
    feature_results.reserve(feature_futures.size());
    uint32_t total_vertices = 0;
    uint32_t total_normals = 0;
    uint32_t total_depths = 0;
    uint32_t total_indices = 0;
    size_t total_keys = 0;
    for (uint32_t i = 0; i < feature_futures.size(); ++i) {
        CaveFeatureExtractionResult result = feature_futures[i].get();
        total_vertices += static_cast<uint32_t>(result.surface.vertices.size());
        total_normals += static_cast<uint32_t>(result.surface.normals.size());
        total_depths += result.surface.vertex_depths.size() == result.surface.vertices.size()
            ? static_cast<uint32_t>(result.surface.vertex_depths.size())
            : static_cast<uint32_t>(result.surface.vertices.size());
        total_indices += static_cast<uint32_t>(result.surface.triangle_indices.size());
        total_keys += result.sign_changing_keys.size();
        mesh.perf.cave_features.push_back(result.stats);
        mesh.surface_net.local_vertex_count += result.stats.vertices;
        mesh.surface_net.local_triangle_count += result.stats.triangles;
        mesh.surface_net.occupied_voxel_count += result.stats.occupied_voxels;
        mesh.surface_net.candidate_cube_count += result.stats.candidate_cubes;
        feature_results.push_back(std::move(result));
        report_index_progress(config, 0.56, 0.84, i + 1u, mesh.voxel_features.size(), "Extracting cave surface nets");
    }

    mesh.surface_net.vertices.reserve(mesh.surface_net.vertices.size() + total_vertices);
    mesh.surface_net.normals.reserve(mesh.surface_net.normals.size() + total_normals);
    mesh.surface_net.vertex_depths.reserve(mesh.surface_net.vertex_depths.size() + total_depths);
    mesh.surface_net.triangle_indices.reserve(mesh.surface_net.triangle_indices.size() + total_indices);
    cave_surface_keys.reserve(total_keys);

    for (const CaveFeatureExtractionResult& result : feature_results) {
        cave_surface_keys.insert(
            cave_surface_keys.end(),
            result.sign_changing_keys.begin(),
            result.sign_changing_keys.end()
        );
        append_surface_net_mesh(mesh.surface_net, result.surface);
    }
    mesh.perf.cave_surface_net_ms = elapsed_ms(cave_surface_begin);
    mesh.perf.cave_surface_vertices = mesh.surface_net.local_vertex_count;
    mesh.perf.cave_surface_triangles = mesh.surface_net.local_triangle_count;

    sort_and_unique_voxel_keys(cave_surface_keys);
    mesh.svo.bounds_radius = grid_radius;
    mesh.svo.depth = depth;
    mesh.svo.max_depth = depth;
    mesh.svo.debug_draw_depth = std::min(config.svo_debug_draw_depth, depth);
    mesh.svo.debug_max_boxes = std::max(1u, config.svo_debug_max_boxes);
    mesh.svo.occupied_leaf_count = static_cast<uint32_t>(cave_surface_keys.size());
    mesh.svo.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    mesh.svo.local_edit_depth = config.voxel_edits.digs.empty() ? 0u : config.voxel_edits.local_depth;
    if (!cave_surface_keys.empty()) {
        report_progress(config, 0.88f, "Building cave SVO debug tree");
        const auto cave_svo_begin = PerfClock::now();
        const uint32_t resolution = 1u << depth;
        const uint32_t debug_tree_depth = std::min(depth, mesh.svo.debug_draw_depth);
        mesh.svo.nodes.resize(1);
        build_svo_node_at(
            mesh.svo.nodes,
            0,
            cave_surface_keys,
            0,
            static_cast<uint32_t>(cave_surface_keys.size()),
            0,
            0,
            0,
            0,
            resolution,
            debug_tree_depth
        );
        count_svo_debug_boxes_recursive(mesh.svo, 0, debug_tree_depth, mesh.svo.debug_max_boxes, mesh.svo.debug_box_count);
        mesh.perf.cave_svo_ms = elapsed_ms(cave_svo_begin);
    }
    mesh.perf.voxel_total_ms = elapsed_ms(cave_begin);
    report_progress(config, 0.97f, "Cave voxel data ready");
}

void bias_surface_net_outward(SurfaceNetMesh& surface_net, float bias) {
    if (bias <= 0.0f) {
        return;
    }

    for (Vec3& vertex : surface_net.vertices) {
        if (length(vertex) <= 0.000001f) {
            continue;
        }
        vertex = vertex + normalize(vertex) * bias;
    }
}

bool surface_net_position_in_patch(Vec3 position, const std::vector<LocalSurfaceNetPatch>& patches, float pad) {
    for (const LocalSurfaceNetPatch& patch : patches) {
        if (patch.dig_radius_mesh <= 0.0f) {
            continue;
        }
        if (length(position - patch.dig_center_mesh) <= patch.dig_radius_mesh + pad) {
            return true;
        }
    }
    return false;
}

bool surface_net_position_in_replacement_patch(Vec3 position, const std::vector<LocalSurfaceNetPatch>& patches, float pad) {
    for (const LocalSurfaceNetPatch& patch : patches) {
        if (!patch.replace_surface) {
            continue;
        }
        if (length(position - patch.center_mesh) <= patch.suppress_radius_mesh + pad) {
            return true;
        }
    }
    return false;
}

bool point_in_triangle_2d(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
    const Vec2 v0 = c - a;
    const Vec2 v1 = b - a;
    const Vec2 v2 = p - a;
    const float dot00 = dot2(v0, v0);
    const float dot01 = dot2(v0, v1);
    const float dot02 = dot2(v0, v2);
    const float dot11 = dot2(v1, v1);
    const float dot12 = dot2(v1, v2);
    const float denominator = dot00 * dot11 - dot01 * dot01;
    if (std::fabs(denominator) <= 0.000000000001f) {
        return false;
    }
    const float inv_denominator = 1.0f / denominator;
    const float u = (dot11 * dot02 - dot01 * dot12) * inv_denominator;
    const float v = (dot00 * dot12 - dot01 * dot02) * inv_denominator;
    return u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f;
}

bool surface_net_triangle_overlaps_patch_radius(
    Vec3 a,
    Vec3 b,
    Vec3 c,
    Vec3 center,
    float radius
) {
    if (radius <= 0.0f || length(center) <= 0.000001f) {
        return false;
    }

    const Vec3 axis = normalize(center);
    const Vec3 reference = std::fabs(axis.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference, axis));
    const Vec3 bitangent = cross(axis, tangent);
    const float radius_sq = radius * radius;

    auto project = [&](Vec3 position) {
        const Vec3 relative = position - center;
        return Vec2{dot(relative, tangent), dot(relative, bitangent)};
    };

    const Vec2 pa = project(a);
    const Vec2 pb = project(b);
    const Vec2 pc = project(c);
    const Vec2 origin{0.0f, 0.0f};
    if (dot2(pa, pa) <= radius_sq || dot2(pb, pb) <= radius_sq || dot2(pc, pc) <= radius_sq) {
        return true;
    }
    if (point_segment_distance(origin, pa, pb) <= radius ||
        point_segment_distance(origin, pb, pc) <= radius ||
        point_segment_distance(origin, pc, pa) <= radius) {
        return true;
    }
    return point_in_triangle_2d(origin, pa, pb, pc);
}

bool surface_net_triangle_overlaps_dig_patch(
    Vec3 a,
    Vec3 b,
    Vec3 c,
    const LocalSurfaceNetPatch& patch,
    float pad
) {
    return surface_net_triangle_overlaps_patch_radius(
        a,
        b,
        c,
        patch.dig_center_mesh,
        patch.dig_radius_mesh + std::max(0.0f, pad)
    );
}

bool surface_net_triangle_overlaps_dig_patches(
    Vec3 a,
    Vec3 b,
    Vec3 c,
    const std::vector<LocalSurfaceNetPatch>& patches,
    float pad
) {
    for (const LocalSurfaceNetPatch& patch : patches) {
        if (surface_net_triangle_overlaps_dig_patch(a, b, c, patch, pad)) {
            return true;
        }
    }
    return false;
}

bool surface_net_triangle_overlaps_replacement_patches(
    Vec3 a,
    Vec3 b,
    Vec3 c,
    const std::vector<LocalSurfaceNetPatch>& patches,
    float pad
) {
    for (const LocalSurfaceNetPatch& patch : patches) {
        if (!patch.replace_surface) {
            continue;
        }
        if (surface_net_triangle_overlaps_patch_radius(
                a,
                b,
                c,
                patch.center_mesh,
                patch.suppress_radius_mesh + std::max(0.0f, pad))) {
            return true;
        }
    }
    return false;
}

Vec3 patch_surface_point(
    Vec3 axis,
    Vec3 tangent,
    Vec3 bitangent,
    float surface_radius,
    float local_radius,
    float angle
) {
    const Vec3 tangent_offset = tangent * (std::cos(angle) * local_radius) +
                                bitangent * (std::sin(angle) * local_radius);
    return normalize(axis * surface_radius + tangent_offset) * surface_radius;
}

void append_surface_net_triangle(
    SurfaceNetMesh& surface_net,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    Vec3 preferred_normal,
    uint32_t max_vertices,
    uint32_t depth_tag = 0u
) {
    if (surface_net.vertices.size() + 3u > max_vertices) {
        return;
    }

    Vec3 normal = cross(b - a, c - a);
    if (length(normal) <= 0.000000000001f) {
        return;
    }
    normal = normalize(normal);
    if (dot(normal, preferred_normal) < 0.0f) {
        std::swap(b, c);
        normal = -normal;
    }

    const uint32_t base = static_cast<uint32_t>(surface_net.vertices.size());
    surface_net.vertices.push_back(a);
    surface_net.vertices.push_back(b);
    surface_net.vertices.push_back(c);
    surface_net.normals.push_back(normal);
    surface_net.normals.push_back(normal);
    surface_net.normals.push_back(normal);
    const uint32_t resolved_depth = depth_tag != 0u ? depth_tag : surface_net.source_depth;
    surface_net.vertex_depths.push_back(resolved_depth);
    surface_net.vertex_depths.push_back(resolved_depth);
    surface_net.vertex_depths.push_back(resolved_depth);
    surface_net.triangle_indices.push_back(base);
    surface_net.triangle_indices.push_back(base + 1u);
    surface_net.triangle_indices.push_back(base + 2u);
}

Vec3 terrain_mask_surface_point(const TerrainHeightMask& mask, float px, float py, float surface_radius) {
    const float resolution_minus_one = static_cast<float>(std::max(1u, mask.resolution - 1u));
    const float clamped_x = std::clamp(px, 0.0f, resolution_minus_one);
    const float clamped_y = std::clamp(py, 0.0f, resolution_minus_one);
    const float radius_mesh = kilometers_to_world_units(mask.radius_km);
    const float local_x = (clamped_x / resolution_minus_one - 0.5f) * radius_mesh * 2.0f;
    const float local_y = (clamped_y / resolution_minus_one - 0.5f) * radius_mesh * 2.0f;
    return normalize(normalize(mask.center_mesh) + mask.tangent_mesh * local_x + mask.bitangent_mesh * local_y) * surface_radius;
}

bool terrain_mask_pixel_is_hole(const TerrainHeightMask& mask, int32_t x, int32_t y) {
    if (x < 0 || y < 0 || x >= static_cast<int32_t>(mask.resolution) || y >= static_cast<int32_t>(mask.resolution)) {
        return false;
    }
    return mask.heights[static_cast<size_t>(y) * mask.resolution + static_cast<uint32_t>(x)] == 0u;
}

void append_terrain_mask_svo_fill(
    SurfaceNetMesh& surface_net,
    const LocalSurfaceNetPatch& patch,
    const MarchingCubesConfig& config,
    float surface_radius,
    float wall_depth,
    uint32_t depth_tag
) {
    if (!patch.replace_surface || config.voxel_edits.terrain_masks.empty() || wall_depth <= 0.0f) {
        return;
    }

    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    const float fill_depth = std::min(wall_depth, kilometers_to_world_units(6.0f));
    const float top_radius = surface_radius + kilometers_to_world_units(0.05f);
    const float bottom_radius = std::max(0.0f, surface_radius - fill_depth);

    auto append_quad = [&](Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3 preferred_normal) {
        append_surface_net_triangle(surface_net, a, b, c, preferred_normal, max_vertices, depth_tag);
        append_surface_net_triangle(surface_net, a, c, d, preferred_normal, max_vertices, depth_tag);
    };

    for (const TerrainHeightMask& mask : config.voxel_edits.terrain_masks) {
        Vec3 footprint_center{};
        float footprint_radius = 0.0f;
        if (!terrain_height_mask_hole_footprint(mask, footprint_center, footprint_radius)) {
            continue;
        }
        if (length(footprint_center - patch.center_mesh) > patch.extraction_radius_mesh + footprint_radius) {
            continue;
        }

        const int32_t resolution = static_cast<int32_t>(mask.resolution);
        for (int32_t y = 0; y < resolution; ++y) {
            for (int32_t x = 0; x < resolution; ++x) {
                if (!terrain_mask_pixel_is_hole(mask, x, y)) {
                    continue;
                }

                const float x0 = static_cast<float>(x) - 0.5f;
                const float x1 = static_cast<float>(x) + 0.5f;
                const float y0 = static_cast<float>(y) - 0.5f;
                const float y1 = static_cast<float>(y) + 0.5f;

                struct Edge {
                    int32_t nx;
                    int32_t ny;
                    float ax;
                    float ay;
                    float bx;
                    float by;
                };
                constexpr std::array<Edge, 4> Edges = {{
                    {-1, 0, -0.5f, -0.5f, -0.5f,  0.5f},
                    { 1, 0,  0.5f,  0.5f,  0.5f, -0.5f},
                    {0, -1,  0.5f, -0.5f, -0.5f, -0.5f},
                    {0,  1, -0.5f,  0.5f,  0.5f,  0.5f},
                }};

                for (const Edge& edge : Edges) {
                    if (terrain_mask_pixel_is_hole(mask, x + edge.nx, y + edge.ny)) {
                        continue;
                    }
                    const Vec3 top_a = terrain_mask_surface_point(mask, static_cast<float>(x) + edge.ax, static_cast<float>(y) + edge.ay, top_radius);
                    const Vec3 top_b = terrain_mask_surface_point(mask, static_cast<float>(x) + edge.bx, static_cast<float>(y) + edge.by, top_radius);
                    const Vec3 bottom_a = terrain_mask_surface_point(mask, static_cast<float>(x) + edge.ax, static_cast<float>(y) + edge.ay, bottom_radius);
                    const Vec3 bottom_b = terrain_mask_surface_point(mask, static_cast<float>(x) + edge.bx, static_cast<float>(y) + edge.by, bottom_radius);
                    const Vec3 wall_normal = normalize(cross(top_b - top_a, bottom_a - top_a));
                    append_quad(top_a, top_b, bottom_b, bottom_a, wall_normal);
                }
            }
        }
    }
}

void append_local_surface_net_transition_fill(
    SurfaceNetMesh& surface_net,
    const LocalSurfaceNetPatch& patch,
    float grid_radius,
    const MarchingCubesConfig& config
) {
    if (!patch.replace_surface || length(patch.center_mesh) <= 0.000001f) {
        return;
    }

    const float outer_radius = patch.suppress_radius_mesh;
    if (outer_radius <= 0.0f) {
        return;
    }

    const float surface_radius = std::max(1.0f, length(patch.center_mesh));
    const float dig_radius = patch.dig_radius_mesh > 0.0f
        ? patch.dig_radius_mesh + kilometers_to_world_units(0.35f)
        : 0.0f;
    const float inner_radius = std::min(dig_radius, outer_radius * 0.9f);
    if (outer_radius <= inner_radius + 0.000001f) {
        return;
    }

    const Vec3 axis = normalize(patch.center_mesh);
    const Vec3 reference = std::fabs(axis.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference, axis));
    const Vec3 bitangent = cross(axis, tangent);

    const uint32_t patch_depth = std::clamp(patch.depth, 1u, MaxSvoDepth);
    const float depth_voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << patch_depth);
    const float surface_bias = std::max(depth_voxel_size * 2.0f, kilometers_to_world_units(0.5f));
    const float fill_surface_radius = surface_radius + surface_bias;
    const float ring_spacing = std::max(depth_voxel_size * 12.0f, kilometers_to_world_units(1.0f));
    const float angular_spacing = std::max(depth_voxel_size * 10.0f, kilometers_to_world_units(1.25f));
    const uint32_t radial_segments = std::clamp(
        static_cast<uint32_t>(std::ceil((outer_radius - inner_radius) / ring_spacing)),
        4u,
        96u
    );
    const uint32_t angular_segments = std::clamp(
        static_cast<uint32_t>(std::ceil((2.0f * Pi * outer_radius) / angular_spacing)),
        32u,
        256u
    );

    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    const Vec3 outward = axis;

    if (inner_radius <= 0.000001f) {
        const Vec3 center = axis * fill_surface_radius;
        const float first_radius = outer_radius / static_cast<float>(radial_segments);
        for (uint32_t i = 0; i < angular_segments; ++i) {
            const float a0 = 2.0f * Pi * static_cast<float>(i) / static_cast<float>(angular_segments);
            const float a1 = 2.0f * Pi * static_cast<float>(i + 1u) / static_cast<float>(angular_segments);
            append_surface_net_triangle(
                surface_net,
                center,
                patch_surface_point(axis, tangent, bitangent, fill_surface_radius, first_radius, a0),
                patch_surface_point(axis, tangent, bitangent, fill_surface_radius, first_radius, a1),
                outward,
                max_vertices
            );
        }
    }

    const uint32_t start_ring = inner_radius <= 0.000001f ? 1u : 0u;
    for (uint32_t ring = start_ring; ring < radial_segments; ++ring) {
        const float t0 = static_cast<float>(ring) / static_cast<float>(radial_segments);
        const float t1 = static_cast<float>(ring + 1u) / static_cast<float>(radial_segments);
        const float r0 = inner_radius + (outer_radius - inner_radius) * t0;
        const float r1 = inner_radius + (outer_radius - inner_radius) * t1;
        for (uint32_t i = 0; i < angular_segments; ++i) {
            const float a0 = 2.0f * Pi * static_cast<float>(i) / static_cast<float>(angular_segments);
            const float a1 = 2.0f * Pi * static_cast<float>(i + 1u) / static_cast<float>(angular_segments);
            const Vec3 p00 = patch_surface_point(axis, tangent, bitangent, fill_surface_radius, r0, a0);
            const Vec3 p01 = patch_surface_point(axis, tangent, bitangent, fill_surface_radius, r0, a1);
            const Vec3 p10 = patch_surface_point(axis, tangent, bitangent, fill_surface_radius, r1, a0);
            const Vec3 p11 = patch_surface_point(axis, tangent, bitangent, fill_surface_radius, r1, a1);
            append_surface_net_triangle(surface_net, p00, p10, p11, outward, max_vertices);
            append_surface_net_triangle(surface_net, p00, p11, p01, outward, max_vertices);
        }
    }
}

void append_local_surface_net_footprint_fill(
    SurfaceNetMesh& surface_net,
    const LocalSurfaceNetPatch& patch,
    float grid_radius,
    const MarchingCubesConfig& config
) {
    if (!patch.replace_surface || length(patch.center_mesh) <= 0.000001f || patch.suppress_radius_mesh <= 0.0f) {
        return;
    }

    const Vec3 axis = normalize(patch.center_mesh);
    const Vec3 reference = std::fabs(axis.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference, axis));
    const Vec3 bitangent = cross(axis, tangent);
    const uint32_t patch_depth = std::clamp(patch.depth, 1u, MaxSvoDepth);
    const float depth_voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << patch_depth);
    const float surface_bias = std::max(depth_voxel_size * 3.0f, kilometers_to_world_units(0.75f));
    const float surface_radius = std::max(1.0f, length(patch.center_mesh)) + surface_bias;
    const float half_extent = patch.suppress_radius_mesh;
    const float dig_radius = patch.dig_radius_mesh > 0.0f
        ? patch.dig_radius_mesh + kilometers_to_world_units(0.25f)
        : 0.0f;
    const float cell_spacing = std::max(depth_voxel_size * 14.0f, kilometers_to_world_units(1.5f));
    const uint32_t segments = std::clamp(
        static_cast<uint32_t>(std::ceil((half_extent * 2.0f) / cell_spacing)),
        8u,
        192u
    );
    const float step = (half_extent * 2.0f) / static_cast<float>(segments);
    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);

    auto sample = [&](float x, float y) {
        const Vec3 offset = tangent * x + bitangent * y;
        return normalize(axis * surface_radius + offset) * surface_radius;
    };

    for (uint32_t y = 0; y < segments; ++y) {
        const float y0 = -half_extent + static_cast<float>(y) * step;
        const float y1 = y0 + step;
        for (uint32_t x = 0; x < segments; ++x) {
            const float x0 = -half_extent + static_cast<float>(x) * step;
            const float x1 = x0 + step;
            const float cx = (x0 + x1) * 0.5f;
            const float cy = (y0 + y1) * 0.5f;
            if (dig_radius > 0.0f && std::sqrt(cx * cx + cy * cy) <= dig_radius) {
                continue;
            }

            const Vec3 p00 = sample(x0, y0);
            const Vec3 p01 = sample(x0, y1);
            const Vec3 p10 = sample(x1, y0);
            const Vec3 p11 = sample(x1, y1);
            append_surface_net_triangle(surface_net, p00, p10, p11, axis, max_vertices);
            append_surface_net_triangle(surface_net, p00, p11, p01, axis, max_vertices);
        }
    }
}

void append_depth_transition_rect(
    SurfaceNetMesh& surface_net,
    Vec3 axis,
    Vec3 tangent,
    Vec3 bitangent,
    float surface_radius,
    float u0,
    float u1,
    float v0,
    float v1,
    float step,
    const LocalSurfaceNetPatch& patch,
    uint32_t max_vertices,
    uint32_t depth_tag
) {
    if (u1 <= u0 || v1 <= v0 || step <= 0.0f) {
        return;
    }

    auto sample = [&](float u, float v) {
        return normalize(axis * surface_radius + tangent * u + bitangent * v) * surface_radius;
    };
    auto blocked_by_dig = [&](Vec3 position) {
        if (patch.dig_radius_mesh <= 0.0f || length(patch.dig_center_mesh) <= 0.000001f) {
            return false;
        }
        const Vec3 dig_axis = normalize(patch.dig_center_mesh);
        const float axial = dot(position - patch.dig_center_mesh, dig_axis);
        const Vec3 axis_point = patch.dig_center_mesh + dig_axis * axial;
        const float tangent_distance = length(position - axis_point);
        return axial <= patch.dig_radius_mesh * 0.65f &&
               tangent_distance <= patch.dig_radius_mesh + step * 3.0f;
    };

    const uint32_t u_segments = std::clamp(
        static_cast<uint32_t>(std::ceil((u1 - u0) / step)),
        1u,
        160u
    );
    const uint32_t v_segments = std::clamp(
        static_cast<uint32_t>(std::ceil((v1 - v0) / step)),
        1u,
        160u
    );

    for (uint32_t y = 0; y < v_segments; ++y) {
        const float ty0 = static_cast<float>(y) / static_cast<float>(v_segments);
        const float ty1 = static_cast<float>(y + 1u) / static_cast<float>(v_segments);
        const float va = v0 + (v1 - v0) * ty0;
        const float vb = v0 + (v1 - v0) * ty1;
        for (uint32_t x = 0; x < u_segments; ++x) {
            const float tx0 = static_cast<float>(x) / static_cast<float>(u_segments);
            const float tx1 = static_cast<float>(x + 1u) / static_cast<float>(u_segments);
            const float ua = u0 + (u1 - u0) * tx0;
            const float ub = u0 + (u1 - u0) * tx1;
            const Vec3 p00 = sample(ua, va);
            const Vec3 p10 = sample(ub, va);
            const Vec3 p01 = sample(ua, vb);
            const Vec3 p11 = sample(ub, vb);
            const Vec3 centroid = (p00 + p10 + p01 + p11) * 0.25f;
            if (blocked_by_dig(centroid)) {
                continue;
            }

            append_surface_net_triangle(surface_net, p00, p10, p11, axis, max_vertices, depth_tag);
            append_surface_net_triangle(surface_net, p00, p11, p01, axis, max_vertices, depth_tag);
        }
    }
}

void append_depth8_to_depth16_transition_skirt(
    SurfaceNetMesh& surface_net,
    const LocalSurfaceNetPatch& patch,
    float grid_radius,
    const MarchingCubesConfig& config
) {
    if (!patch.replace_surface || patch.promoted_depth8_keys.empty() || length(patch.center_mesh) <= 0.000001f) {
        return;
    }

    const Vec3 axis = normalize(patch.center_mesh);
    const Vec3 reference = std::fabs(axis.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference, axis));
    const Vec3 bitangent = cross(axis, tangent);

    float min_u = 1000000.0f;
    float max_u = -1000000.0f;
    float min_v = 1000000.0f;
    float max_v = -1000000.0f;
    for (uint32_t corner = 0; corner < 8u; ++corner) {
        const Vec3 p{
            (corner & 1u) != 0u ? patch.replacement_max.x : patch.replacement_min.x,
            (corner & 2u) != 0u ? patch.replacement_max.y : patch.replacement_min.y,
            (corner & 4u) != 0u ? patch.replacement_max.z : patch.replacement_min.z,
        };
        const Vec3 relative = p - patch.center_mesh;
        const float u = dot(relative, tangent);
        const float v = dot(relative, bitangent);
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    if (max_u <= min_u || max_v <= min_v) {
        return;
    }

    const float root_voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << SurfacePromotionRootDepth);
    const uint32_t patch_depth = std::clamp(patch.depth, 1u, MaxSvoDepth);
    const float detail_voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << patch_depth);
    const float inner_overlap = std::max(detail_voxel_size * 8.0f, kilometers_to_world_units(1.0f));
    const float outer_width = std::max(root_voxel_size * 0.45f, kilometers_to_world_units(18.0f));
    const float step = std::max(detail_voxel_size * 32.0f, kilometers_to_world_units(3.0f));
    const float surface_radius = std::max(1.0f, length(patch.dig_center_mesh)) +
                                 std::max(detail_voxel_size * 4.0f, kilometers_to_world_units(0.75f));
    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    const uint32_t transition_depth_tag = patch_depth > 1u ? patch_depth - 1u : patch_depth;

    const float outer_min_u = min_u - outer_width;
    const float outer_max_u = max_u + outer_width;
    const float outer_min_v = min_v - outer_width;
    const float outer_max_v = max_v + outer_width;

    append_depth_transition_rect(
        surface_net,
        axis,
        tangent,
        bitangent,
        surface_radius,
        outer_min_u,
        min_u + inner_overlap,
        outer_min_v,
        outer_max_v,
        step,
        patch,
        max_vertices,
        transition_depth_tag
    );
    append_depth_transition_rect(
        surface_net,
        axis,
        tangent,
        bitangent,
        surface_radius,
        max_u - inner_overlap,
        outer_max_u,
        outer_min_v,
        outer_max_v,
        step,
        patch,
        max_vertices,
        transition_depth_tag
    );
    append_depth_transition_rect(
        surface_net,
        axis,
        tangent,
        bitangent,
        surface_radius,
        min_u + inner_overlap,
        max_u - inner_overlap,
        outer_min_v,
        min_v + inner_overlap,
        step,
        patch,
        max_vertices,
        transition_depth_tag
    );
    append_depth_transition_rect(
        surface_net,
        axis,
        tangent,
        bitangent,
        surface_radius,
        min_u + inner_overlap,
        max_u - inner_overlap,
        max_v - inner_overlap,
        outer_max_v,
        step,
        patch,
        max_vertices,
        transition_depth_tag
    );
}

void suppress_surface_net_triangles_for_patches(
    SurfaceNetMesh& surface_net,
    const std::vector<LocalSurfaceNetPatch>& patches,
    float pad,
    bool use_replacement_region
) {
    if (patches.empty() || surface_net.triangle_indices.empty()) {
        return;
    }

    std::vector<uint32_t> filtered_indices;
    filtered_indices.reserve(surface_net.triangle_indices.size());
    for (uint32_t i = 0; i + 2 < surface_net.triangle_indices.size(); i += 3) {
        const uint32_t ia = surface_net.triangle_indices[i];
        const uint32_t ib = surface_net.triangle_indices[i + 1u];
        const uint32_t ic = surface_net.triangle_indices[i + 2u];
        if (ia >= surface_net.vertices.size() || ib >= surface_net.vertices.size() || ic >= surface_net.vertices.size()) {
            continue;
        }

        const Vec3 a = surface_net.vertices[ia];
        const Vec3 b = surface_net.vertices[ib];
        const Vec3 c = surface_net.vertices[ic];
        const Vec3 centroid = (a + b + c) / 3.0f;
        const bool remove_triangle = use_replacement_region
            ? (surface_net_position_in_replacement_patch(centroid, patches, pad) ||
               surface_net_triangle_overlaps_replacement_patches(a, b, c, patches, pad))
            : (surface_net_position_in_patch(centroid, patches, pad) ||
               surface_net_triangle_overlaps_dig_patches(a, b, c, patches, pad));
        if (remove_triangle) {
            continue;
        }

        filtered_indices.push_back(ia);
        filtered_indices.push_back(ib);
        filtered_indices.push_back(ic);
    }
    surface_net.triangle_indices = std::move(filtered_indices);
}

bool local_voxel_key_for_position(Vec3 position, Vec3 origin, float voxel_size, uint32_t resolution, VoxelKey& key) {
    auto key_component = [&](float value, float offset, uint32_t& out) {
        const float normalized = (value - offset) / voxel_size;
        const int32_t index = static_cast<int32_t>(std::floor(normalized));
        if (index < 0 || index >= static_cast<int32_t>(resolution)) {
            return false;
        }
        out = static_cast<uint32_t>(index);
        return true;
    };

    return key_component(position.x, origin.x, key.x) &&
           key_component(position.y, origin.y, key.y) &&
           key_component(position.z, origin.z, key.z);
}

Vec3 local_surface_net_sample_position(uint32_t x, uint32_t y, uint32_t z, Vec3 origin, float voxel_size) {
    return {
        origin.x + (static_cast<float>(x) + 0.5f) * voxel_size,
        origin.y + (static_cast<float>(y) + 0.5f) * voxel_size,
        origin.z + (static_cast<float>(z) + 0.5f) * voxel_size,
    };
}

bool local_surface_net_sample_carved(
    Vec3 position,
    const std::vector<PreparedVoxelDig>& digs,
    const std::vector<TerrainHeightMask>& masks
) {
    if (terrain_height_masks_carve(position, masks)) {
        return true;
    }
    for (const PreparedVoxelDig& dig : digs) {
        if (length(dig.center_mesh) <= 0.000001f) {
            continue;
        }
        const Vec3 axis = normalize(dig.center_mesh);
        const float axial = dot(position - dig.center_mesh, axis);
        const Vec3 axis_point = dig.center_mesh + axis * axial;
        const float tangent_distance = length(position - axis_point);
        if (axial <= dig.radius_mesh * 0.35f && tangent_distance <= dig.radius_mesh) {
            return true;
        }
    }
    return false;
}

bool local_surface_net_near_dig_boundary(Vec3 position, const std::vector<PreparedVoxelDig>& digs, float voxel_size) {
    const float tolerance = voxel_size * 3.0f;
    for (const PreparedVoxelDig& dig : digs) {
        if (length(dig.center_mesh) <= 0.000001f) {
            continue;
        }
        const Vec3 axis = normalize(dig.center_mesh);
        const float axial = dot(position - dig.center_mesh, axis);
        const Vec3 axis_point = dig.center_mesh + axis * axial;
        const float tangent_distance = length(position - axis_point);
        if (axial <= dig.radius_mesh * 0.5f && std::fabs(tangent_distance - dig.radius_mesh) <= tolerance) {
            return true;
        }
    }
    return false;
}

void push_local_voxel_key(
    std::vector<VoxelKey>& keys,
    Vec3 position,
    const LocalSurfaceNetPatch& patch,
    Vec3 origin,
    float voxel_size,
    uint32_t resolution,
    const std::vector<PreparedVoxelDig>& digs,
    const std::vector<TerrainHeightMask>& masks
) {
    if (length(position - patch.center_mesh) > patch.extraction_radius_mesh || local_surface_net_sample_carved(position, digs, masks)) {
        return;
    }

    VoxelKey key;
    if (local_voxel_key_for_position(position, origin, voxel_size, resolution, key)) {
        keys.push_back(key);
    }
}

void add_local_voxelized_triangles(
    std::vector<VoxelKey>& keys,
    const QuantizedMesh& mesh,
    const std::vector<uint32_t>& indices,
    const LocalSurfaceNetPatch& patch,
    Vec3 origin,
    float voxel_size,
    uint32_t resolution,
    const std::vector<PreparedVoxelDig>& digs,
    const std::vector<TerrainHeightMask>& masks
) {
    const Vec3 patch_min = origin;
    const Vec3 patch_max = origin + Vec3{
        voxel_size * static_cast<float>(resolution),
        voxel_size * static_cast<float>(resolution),
        voxel_size * static_cast<float>(resolution),
    };
    const float sample_spacing = std::max(voxel_size * 1.25f, 0.00001f);

    for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vec3 a = mesh.vertices[indices[i]].position;
        const Vec3 b = mesh.vertices[indices[i + 1]].position;
        const Vec3 c = mesh.vertices[indices[i + 2]].position;
        const Vec3 tri_min = {
            std::min(a.x, std::min(b.x, c.x)),
            std::min(a.y, std::min(b.y, c.y)),
            std::min(a.z, std::min(b.z, c.z)),
        };
        const Vec3 tri_max = {
            std::max(a.x, std::max(b.x, c.x)),
            std::max(a.y, std::max(b.y, c.y)),
            std::max(a.z, std::max(b.z, c.z)),
        };
        if (tri_max.x < patch_min.x || tri_min.x > patch_max.x ||
            tri_max.y < patch_min.y || tri_min.y > patch_max.y ||
            tri_max.z < patch_min.z || tri_min.z > patch_max.z) {
            continue;
        }

        const float longest_edge = std::max(length(b - a), std::max(length(c - b), length(a - c)));
        const uint32_t steps = std::clamp(static_cast<uint32_t>(std::ceil(longest_edge / sample_spacing)), 1u, 64u);
        for (uint32_t u_step = 0; u_step <= steps; ++u_step) {
            for (uint32_t v_step = 0; v_step + u_step <= steps; ++v_step) {
                const float u = static_cast<float>(u_step) / static_cast<float>(steps);
                const float v = static_cast<float>(v_step) / static_cast<float>(steps);
                const float w = 1.0f - u - v;
                push_local_voxel_key(keys, a * u + b * v + c * w, patch, origin, voxel_size, resolution, digs, masks);
            }
        }
        push_local_voxel_key(keys, (a + b + c) / 3.0f, patch, origin, voxel_size, resolution, digs, masks);
        push_local_voxel_key(keys, (a + b) * 0.5f, patch, origin, voxel_size, resolution, digs, masks);
        push_local_voxel_key(keys, (b + c) * 0.5f, patch, origin, voxel_size, resolution, digs, masks);
        push_local_voxel_key(keys, (c + a) * 0.5f, patch, origin, voxel_size, resolution, digs, masks);
    }
}

void fill_local_radial_occupancy(
    std::vector<uint64_t>& occupancy,
    std::vector<VoxelKey>& occupied_keys,
    const LocalSurfaceNetPatch& patch,
    Vec3 origin,
    float voxel_size,
    uint32_t resolution,
    float surface_radius,
    const std::vector<PreparedVoxelDig>& digs,
    const std::vector<TerrainHeightMask>& masks
) {
    const float shell_radius = patch.extraction_radius_mesh + std::sqrt(3.0f) * voxel_size;
    const bool use_replacement_bounds = !patch.promoted_depth8_keys.empty();
    occupied_keys.clear();
    for (uint32_t z = 0; z < resolution; ++z) {
        for (uint32_t y = 0; y < resolution; ++y) {
            for (uint32_t x = 0; x < resolution; ++x) {
                const Vec3 position = local_surface_net_sample_position(x, y, z, origin, voxel_size);
                if ((!use_replacement_bounds && length(position - patch.center_mesh) > shell_radius) ||
                    length(position) > surface_radius ||
                    local_surface_net_sample_carved(position, digs, masks)) {
                    continue;
                }
                set_surface_net_bit(occupancy, surface_net_grid_index(x, y, z, resolution));
            }
        }
    }

    for (uint32_t z = 1; z + 1 < resolution; ++z) {
        for (uint32_t y = 1; y + 1 < resolution; ++y) {
            for (uint32_t x = 1; x + 1 < resolution; ++x) {
                if (!surface_net_bit(occupancy, surface_net_grid_index(x, y, z, resolution))) {
                    continue;
                }
                const bool boundary =
                    !surface_net_bit(occupancy, surface_net_grid_index(x - 1u, y, z, resolution)) ||
                    !surface_net_bit(occupancy, surface_net_grid_index(x + 1u, y, z, resolution)) ||
                    !surface_net_bit(occupancy, surface_net_grid_index(x, y - 1u, z, resolution)) ||
                    !surface_net_bit(occupancy, surface_net_grid_index(x, y + 1u, z, resolution)) ||
                    !surface_net_bit(occupancy, surface_net_grid_index(x, y, z - 1u, resolution)) ||
                    !surface_net_bit(occupancy, surface_net_grid_index(x, y, z + 1u, resolution));
                const Vec3 position = local_surface_net_sample_position(x, y, z, origin, voxel_size);
                const bool inside_patch =
                    (use_replacement_bounds &&
                        position.x >= patch.replacement_min.x && position.y >= patch.replacement_min.y && position.z >= patch.replacement_min.z &&
                        position.x <= patch.replacement_max.x && position.y <= patch.replacement_max.y && position.z <= patch.replacement_max.z) ||
                    length(position - patch.center_mesh) <= patch.extraction_radius_mesh - voxel_size * 2.0f;
                if (boundary && inside_patch) {
                    occupied_keys.push_back({x, y, z});
                }
            }
        }
    }
}

LocalSurfaceNetGrid local_surface_net_grid_for_patch(
    const LocalSurfaceNetPatch& patch,
    float grid_radius
) {
    LocalSurfaceNetGrid grid;
    grid.depth = std::clamp(patch.depth, 1u, MaxSvoDepth);
    const uint32_t global_resolution = 1u << grid.depth;
    grid.voxel_size = (grid_radius * 2.0f) / static_cast<float>(global_resolution);
    if (!patch.promoted_depth8_keys.empty()) {
        const Vec3 extent = patch.replacement_max - patch.replacement_min;
        const float longest_extent = std::max(extent.x, std::max(extent.y, extent.z));
        grid.resolution = static_cast<uint32_t>(std::ceil(longest_extent / grid.voxel_size)) + 3u;
        grid.resolution = std::clamp(grid.resolution, 4u, LocalSurfaceNetMaxResolution);
        grid.radius = grid.voxel_size * static_cast<float>(grid.resolution) * 0.5f;
        grid.origin = patch.replacement_min - Vec3{grid.voxel_size, grid.voxel_size, grid.voxel_size};
        return grid;
    }
    grid.resolution = static_cast<uint32_t>(std::ceil((patch.extraction_radius_mesh * 2.0f) / grid.voxel_size)) + 4u;
    grid.resolution = std::clamp(grid.resolution, 4u, 192u);
    grid.radius = grid.voxel_size * static_cast<float>(grid.resolution) * 0.5f;
    grid.origin = patch.center_mesh - Vec3{grid.radius, grid.radius, grid.radius};
    return grid;
}

std::vector<LocalSurfaceNetPatch> realized_local_surface_net_patches(
    const std::vector<LocalSurfaceNetPatch>& patches,
    float grid_radius
) {
    std::vector<LocalSurfaceNetPatch> realized;
    realized.reserve(patches.size());
    for (LocalSurfaceNetPatch patch : patches) {
        const LocalSurfaceNetGrid grid = local_surface_net_grid_for_patch(patch, grid_radius);
        const float realized_detail_radius = std::max(
            patch.suppress_radius_mesh,
            std::max(0.0f, grid.radius - grid.voxel_size * 3.0f)
        );
        const float requested_extraction_radius = patch.extraction_radius_mesh;
        patch.extraction_radius_mesh = std::min(requested_extraction_radius, realized_detail_radius);
        patch.suppress_radius_mesh = patch.replace_surface
            ? patch.extraction_radius_mesh
            : patch.suppress_radius_mesh;
        realized.push_back(patch);
    }
    return realized;
}

SurfaceNetMesh build_fast_local_radial_surface_patch_mesh(
    const QuantizedMesh& mesh,
    const LocalSurfaceNetPatch& patch,
    float grid_radius,
    const MarchingCubesConfig& config,
    double progress_begin,
    double progress_end
) {
    SurfaceNetMesh surface_net;
    if (!config.enable_surface_net_generation ||
        patch.promoted_depth8_keys.empty() ||
        length(patch.dig_center_mesh) <= 0.000001f) {
        return surface_net;
    }

    const uint32_t depth = std::clamp(patch.depth, 1u, MaxSvoDepth);
    const float voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << depth);
    const float root_voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << SurfacePromotionRootDepth);
    const Vec3 axis = normalize(patch.dig_center_mesh);
    const Vec3 reference = std::fabs(axis.y) < 0.9f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference, axis));
    const Vec3 bitangent = cross(axis, tangent);
    const float replacement_pad = std::max(root_voxel_size * 0.75f, voxel_size * 8.0f);
    const float fallback_surface_radius = std::max(0.000001f, length(patch.dig_center_mesh));

    float replaced_surface_radius = fallback_surface_radius;
    uint32_t replaced_sample_count = 0;
    for (const Vec3& vertex : mesh.surface_net_base_cache.vertices) {
        if (vertex.x < patch.replacement_min.x - replacement_pad ||
            vertex.y < patch.replacement_min.y - replacement_pad ||
            vertex.z < patch.replacement_min.z - replacement_pad ||
            vertex.x > patch.replacement_max.x + replacement_pad ||
            vertex.y > patch.replacement_max.y + replacement_pad ||
            vertex.z > patch.replacement_max.z + replacement_pad) {
            continue;
        }

        const float radius = length(vertex);
        if (radius > replaced_surface_radius) {
            replaced_surface_radius = radius;
        }
        ++replaced_sample_count;
    }
    if (replaced_sample_count == 0u) {
        replaced_surface_radius = std::min(grid_radius, fallback_surface_radius + root_voxel_size * 0.5f);
    }
    replaced_surface_radius = std::clamp(replaced_surface_radius, fallback_surface_radius, grid_radius);

    const Vec3 plane_origin = axis * replaced_surface_radius;

    float min_u = 1000000.0f;
    float max_u = -1000000.0f;
    float min_v = 1000000.0f;
    float max_v = -1000000.0f;
    for (uint32_t corner = 0; corner < 8u; ++corner) {
        const Vec3 p{
            (corner & 1u) != 0u ? patch.replacement_max.x : patch.replacement_min.x,
            (corner & 2u) != 0u ? patch.replacement_max.y : patch.replacement_min.y,
            (corner & 4u) != 0u ? patch.replacement_max.z : patch.replacement_min.z,
        };
        const Vec3 relative = p - plane_origin;
        const float u = dot(relative, tangent);
        const float v = dot(relative, bitangent);
        min_u = std::min(min_u, u);
        max_u = std::max(max_u, u);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    if (max_u <= min_u || max_v <= min_v) {
        return surface_net;
    }

    const float sample_step = std::max(voxel_size * 2.0f, kilometers_to_world_units(0.35f));
    const uint32_t u_segments = std::clamp(
        static_cast<uint32_t>(std::ceil((max_u - min_u) / sample_step)),
        2u,
        320u
    );
    const uint32_t v_segments = std::clamp(
        static_cast<uint32_t>(std::ceil((max_v - min_v) / sample_step)),
        2u,
        320u
    );
    const float bounds_pad = replacement_pad;
    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    const std::vector<PreparedVoxelDig> digs = prepare_exact_voxel_digs(config.voxel_edits);

    surface_net.source_depth = depth;
    surface_net.bounds_radius = grid_radius;
    surface_net.material_id = config.surface_net_material_id;
    if (!config.voxel_edits.terrain_masks.empty()) {
        const float mask_fill_depth = std::max(voxel_size * 4.0f, kilometers_to_world_units(4.0f));
        append_terrain_mask_svo_fill(surface_net, patch, config, replaced_surface_radius, mask_fill_depth, depth);
        surface_net.occupied_voxel_count = static_cast<uint32_t>(surface_net.vertices.size());
        surface_net.candidate_cube_count = static_cast<uint32_t>(surface_net.triangle_indices.size() / 3u);
        return surface_net;
    }

    surface_net.vertices.reserve(std::min<uint32_t>((u_segments + 1u) * (v_segments + 1u), max_vertices));
    surface_net.normals.reserve(surface_net.vertices.capacity());
    surface_net.vertex_depths.reserve(surface_net.vertices.capacity());
    surface_net.triangle_indices.reserve(static_cast<size_t>(u_segments) * static_cast<size_t>(v_segments) * 6u);

    auto sample = [&](float u, float v) {
        return normalize(plane_origin + tangent * u + bitangent * v) * replaced_surface_radius;
    };
    auto sample_carved = [&](Vec3 position) {
        if (terrain_height_masks_carve(position, config.voxel_edits.terrain_masks)) {
            return true;
        }
        for (const PreparedVoxelDig& dig : digs) {
            if (length(dig.center_mesh) <= 0.000001f) {
                continue;
            }
            const Vec3 dig_axis = normalize(dig.center_mesh);
            const float axial = dot(position - dig.center_mesh, dig_axis);
            const Vec3 axis_point = dig.center_mesh + dig_axis * axial;
            const float tangent_distance = length(position - axis_point);
            const float allowed_axial = std::max(dig.radius_mesh * 0.35f, root_voxel_size * 1.25f);
            if (axial <= allowed_axial && tangent_distance <= dig.radius_mesh) {
                return true;
            }
        }
        return false;
    };
    auto inside_replacement_bounds = [&](Vec3 p) {
        return p.x >= patch.replacement_min.x - bounds_pad &&
               p.y >= patch.replacement_min.y - bounds_pad &&
               p.z >= patch.replacement_min.z - bounds_pad &&
               p.x <= patch.replacement_max.x + bounds_pad &&
               p.y <= patch.replacement_max.y + bounds_pad &&
               p.z <= patch.replacement_max.z + bounds_pad;
    };

    std::vector<int32_t> grid_vertices(static_cast<size_t>(u_segments + 1u) * static_cast<size_t>(v_segments + 1u), -1);
    auto grid_index = [&](uint32_t x, uint32_t y) {
        return static_cast<size_t>(y) * static_cast<size_t>(u_segments + 1u) + static_cast<size_t>(x);
    };

    const double vertex_progress_end = progress_begin + (progress_end - progress_begin) * 0.70;
    const double quad_progress_end = progress_begin + (progress_end - progress_begin) * 0.88;
    const uint64_t vertex_total = static_cast<uint64_t>(u_segments + 1u) * static_cast<uint64_t>(v_segments + 1u);
    uint64_t vertex_processed = 0;
    report_progress(config, progress_begin, "Placing local surface vertices");
    for (uint32_t y = 0; y <= v_segments; ++y) {
        const float ty = static_cast<float>(y) / static_cast<float>(v_segments);
        const float v = min_v + (max_v - min_v) * ty;
        for (uint32_t x = 0; x <= u_segments; ++x) {
            ++vertex_processed;
            report_index_progress(config, progress_begin, vertex_progress_end, vertex_processed, vertex_total, nullptr);
            if (surface_net.vertices.size() >= max_vertices) {
                break;
            }
            const float tx = static_cast<float>(x) / static_cast<float>(u_segments);
            const float u = min_u + (max_u - min_u) * tx;
            const Vec3 p = sample(u, v);
            if (!inside_replacement_bounds(p) || sample_carved(p)) {
                continue;
            }

            const uint32_t vertex_index = static_cast<uint32_t>(surface_net.vertices.size());
            surface_net.vertices.push_back(p);
            surface_net.normals.push_back(p);
            surface_net.vertex_depths.push_back(depth);
            grid_vertices[grid_index(x, y)] = static_cast<int32_t>(vertex_index);
        }
    }

    report_progress(config, vertex_progress_end, "Connecting local surface quads");
    const uint64_t quad_total = static_cast<uint64_t>(u_segments) * static_cast<uint64_t>(v_segments);
    uint64_t quad_processed = 0;
    for (uint32_t y = 0; y < v_segments; ++y) {
        for (uint32_t x = 0; x < u_segments; ++x) {
            ++quad_processed;
            report_index_progress(config, vertex_progress_end, quad_progress_end, quad_processed, quad_total, nullptr);
            const int32_t a = grid_vertices[grid_index(x, y)];
            const int32_t b = grid_vertices[grid_index(x + 1u, y)];
            const int32_t c = grid_vertices[grid_index(x + 1u, y + 1u)];
            const int32_t d = grid_vertices[grid_index(x, y + 1u)];
            append_surface_net_quad(surface_net, a, b, c, d);
        }
    }

    const float wall_depth = std::max(root_voxel_size * 0.95f, voxel_size * 8.0f);
    auto cell_has_surface = [&](uint32_t x, uint32_t y) {
        if (x >= u_segments || y >= v_segments) {
            return false;
        }
        return grid_vertices[grid_index(x, y)] >= 0 &&
               grid_vertices[grid_index(x + 1u, y)] >= 0 &&
               grid_vertices[grid_index(x + 1u, y + 1u)] >= 0 &&
               grid_vertices[grid_index(x, y + 1u)] >= 0;
    };
    auto append_wall = [&](int32_t ia, int32_t ib) {
        if (ia < 0 || ib < 0 || surface_net.vertices.size() + 6u > max_vertices) {
            return;
        }
        const Vec3 top_a = surface_net.vertices[static_cast<uint32_t>(ia)];
        const Vec3 top_b = surface_net.vertices[static_cast<uint32_t>(ib)];
        const Vec3 bottom_a = normalize(top_a) * std::max(0.0f, length(top_a) - wall_depth);
        const Vec3 bottom_b = normalize(top_b) * std::max(0.0f, length(top_b) - wall_depth);
        const Vec3 preferred_normal = normalize(cross(top_b - top_a, bottom_a - top_a));
        append_surface_net_triangle(surface_net, top_a, bottom_a, bottom_b, preferred_normal, max_vertices, depth);
        append_surface_net_triangle(surface_net, top_a, bottom_b, top_b, preferred_normal, max_vertices, depth);
    };

    report_progress(config, quad_progress_end, "Building local surface walls");
    const uint64_t wall_total =
        static_cast<uint64_t>(v_segments + 1u) * static_cast<uint64_t>(u_segments) +
        static_cast<uint64_t>(u_segments + 1u) * static_cast<uint64_t>(v_segments);
    uint64_t wall_processed = 0;
    for (uint32_t y = 0; y <= v_segments; ++y) {
        for (uint32_t x = 0; x < u_segments; ++x) {
            ++wall_processed;
            report_index_progress(config, quad_progress_end, progress_end, wall_processed, wall_total, nullptr);
            const bool below = y > 0u && cell_has_surface(x, y - 1u);
            const bool above = y < v_segments && cell_has_surface(x, y);
            if (below == above) {
                continue;
            }
            append_wall(grid_vertices[grid_index(x, y)], grid_vertices[grid_index(x + 1u, y)]);
        }
    }
    for (uint32_t y = 0; y < v_segments; ++y) {
        for (uint32_t x = 0; x <= u_segments; ++x) {
            ++wall_processed;
            report_index_progress(config, quad_progress_end, progress_end, wall_processed, wall_total, nullptr);
            const bool left = x > 0u && cell_has_surface(x - 1u, y);
            const bool right = x < u_segments && cell_has_surface(x, y);
            if (left == right) {
                continue;
            }
            append_wall(grid_vertices[grid_index(x, y)], grid_vertices[grid_index(x, y + 1u)]);
        }
    }

    append_terrain_mask_svo_fill(surface_net, patch, config, replaced_surface_radius, wall_depth, depth);

    surface_net.occupied_voxel_count = static_cast<uint32_t>(surface_net.vertices.size());
    surface_net.candidate_cube_count = static_cast<uint32_t>(u_segments * v_segments);
    return surface_net;
}

SurfaceNetMesh build_local_surface_net_patch_mesh(
    const QuantizedMesh& mesh,
    const LocalSurfaceNetPatch& patch,
    float grid_radius,
    const MarchingCubesConfig& config,
    double progress_begin,
    double progress_end
) {
    SurfaceNetMesh surface_net;
    if (!config.enable_surface_net_generation) {
        return surface_net;
    }

    if (!config.enable_fractures && !patch.promoted_depth8_keys.empty()) {
        return build_fast_local_radial_surface_patch_mesh(mesh, patch, grid_radius, config, progress_begin, progress_end);
    }

    const LocalSurfaceNetGrid grid = local_surface_net_grid_for_patch(patch, grid_radius);
    const uint32_t depth = grid.depth;
    const float voxel_size = grid.voxel_size;
    const uint32_t resolution = grid.resolution;
    const Vec3 origin = grid.origin;

    std::vector<VoxelKey> occupied_keys;
    occupied_keys.reserve(32768);
    const uint32_t grid_count = resolution * resolution * resolution;
    std::vector<uint64_t> occupancy((grid_count + 63u) / 64u, 0u);
    const std::vector<PreparedVoxelDig> digs = prepare_exact_voxel_digs(config.voxel_edits);

    if (!config.enable_fractures) {
        fill_local_radial_occupancy(
            occupancy,
            occupied_keys,
            patch,
            origin,
            voxel_size,
            resolution,
            1.0f,
            digs,
            config.voxel_edits.terrain_masks
        );
    } else {
        add_local_voxelized_triangles(occupied_keys, mesh, mesh.triangle_indices, patch, origin, voxel_size, resolution, digs, config.voxel_edits.terrain_masks);
        add_local_voxelized_triangles(occupied_keys, mesh, mesh.stitch_triangle_indices, patch, origin, voxel_size, resolution, digs, config.voxel_edits.terrain_masks);
    }

    sort_and_unique_voxel_keys(occupied_keys);
    if (occupied_keys.empty()) {
        return surface_net;
    }
    if (config.enable_fractures) {
        for (VoxelKey key : occupied_keys) {
            if (key.x < resolution && key.y < resolution && key.z < resolution) {
                set_surface_net_bit(occupancy, surface_net_grid_index(key.x, key.y, key.z, resolution));
            }
        }
    }

    const uint32_t cube_resolution = resolution - 1u;
    const uint32_t cube_count = cube_resolution * cube_resolution * cube_resolution;
    std::vector<uint64_t> cube_candidate_bits((cube_count + 63u) / 64u, 0u);
    const uint64_t edge_candidate_count = static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * static_cast<uint64_t>(resolution) * 3ull;
    std::vector<uint64_t> edge_candidate_bits(static_cast<size_t>((edge_candidate_count + 63ull) / 64ull), 0u);
    const double candidate_progress_end = progress_begin + (progress_end - progress_begin) * 0.30;
    const double sort_progress_end = progress_begin + (progress_end - progress_begin) * 0.38;
    const double vertex_progress_end = progress_begin + (progress_end - progress_begin) * 0.78;
    const uint64_t occupied_total = occupied_keys.size();
    uint64_t occupied_processed = 0;
    report_progress(config, progress_begin, "Collecting local surface candidates");
    for (VoxelKey key : occupied_keys) {
        ++occupied_processed;
        report_index_progress(config, progress_begin, candidate_progress_end, occupied_processed, occupied_total, nullptr);
        for (uint32_t dz = 0; dz < 2; ++dz) {
            for (uint32_t dy = 0; dy < 2; ++dy) {
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    if ((dx == 1u && key.x == 0u) || (dy == 1u && key.y == 0u) || (dz == 1u && key.z == 0u)) {
                        continue;
                    }
                    const uint32_t cube_x = key.x - dx;
                    const uint32_t cube_y = key.y - dy;
                    const uint32_t cube_z = key.z - dz;
                    set_surface_net_cube_candidate(cube_candidate_bits, cube_resolution, cube_x, cube_y, cube_z);
                }
            }
        }

        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 0u, false);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 0u, true);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 1u, false);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 1u, true);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 2u, false);
        set_surface_net_edge_candidate_bit(edge_candidate_bits, occupancy, resolution, key.x, key.y, key.z, 2u, true);
    }

    report_progress(config, candidate_progress_end, "Compacting local surface candidates");
    const double cube_compact_progress_end = candidate_progress_end + (sort_progress_end - candidate_progress_end) * 0.40;
    std::vector<VoxelKey> cube_candidates = compact_surface_net_cube_candidates(
        cube_candidate_bits,
        cube_resolution,
        config,
        candidate_progress_end,
        cube_compact_progress_end
    );
    std::vector<SurfaceNetEdgeKey> edge_candidates = compact_surface_net_edge_candidates(
        edge_candidate_bits,
        resolution,
        config,
        cube_compact_progress_end,
        sort_progress_end
    );
    report_progress(config, sort_progress_end, "Placing local surface vertices");

    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    std::vector<int32_t> cube_vertices(cube_count, -1);
    surface_net.source_depth = depth;
    surface_net.bounds_radius = grid_radius;
    surface_net.occupied_voxel_count = static_cast<uint32_t>(occupied_keys.size());
    surface_net.candidate_cube_count = static_cast<uint32_t>(cube_candidates.size());
    surface_net.material_id = config.surface_net_material_id;
    surface_net.vertices.reserve(std::min<uint32_t>(surface_net.candidate_cube_count, max_vertices));
    surface_net.normals.reserve(surface_net.vertices.capacity());

    const uint64_t cube_total = cube_candidates.size();
    uint64_t cube_processed = 0;
    for (VoxelKey cube : cube_candidates) {
        ++cube_processed;
        report_index_progress(config, sort_progress_end, vertex_progress_end, cube_processed, cube_total, nullptr);
        if (surface_net.vertices.size() >= max_vertices) {
            break;
        }

        const uint32_t inside_mask = surface_net_dense_corner_mask(occupancy, resolution, cube.x, cube.y, cube.z);
        const uint32_t inside_count = std::popcount(inside_mask);
        if (inside_count == 0u || inside_count == 8u) {
            continue;
        }

        const SurfaceNetPlacement placement = surface_net_mask_placement(
            inside_mask,
            local_surface_net_sample_position(cube.x, cube.y, cube.z, origin, voxel_size),
            voxel_size,
            config.enable_surface_net_dual_contouring
        );
        if (length(placement.normal) <= 0.000001f) {
            continue;
        }

        const Vec3 vertex = placement.position;
        const bool use_replacement_bounds = !patch.promoted_depth8_keys.empty();
        if ((!use_replacement_bounds && length(vertex - patch.center_mesh) > patch.extraction_radius_mesh - voxel_size * 2.0f) ||
            (use_replacement_bounds &&
                (vertex.x < patch.replacement_min.x - voxel_size || vertex.y < patch.replacement_min.y - voxel_size || vertex.z < patch.replacement_min.z - voxel_size ||
                 vertex.x > patch.replacement_max.x + voxel_size || vertex.y > patch.replacement_max.y + voxel_size || vertex.z > patch.replacement_max.z + voxel_size))) {
            continue;
        }
        const uint32_t vertex_index = static_cast<uint32_t>(surface_net.vertices.size());
        surface_net.vertices.push_back(vertex);
        surface_net.normals.push_back(placement.normal);
        surface_net.vertex_depths.push_back(depth);
        cube_vertices[surface_net_cube_index(cube.x, cube.y, cube.z, cube_resolution)] = static_cast<int32_t>(vertex_index);
    }

    report_progress(config, vertex_progress_end, "Connecting local surface quads");
    surface_net.triangle_indices.reserve(edge_candidates.size() * 6u);
    const uint64_t edge_total = edge_candidates.size();
    uint64_t edge_processed = 0;
    for (SurfaceNetEdgeKey edge : edge_candidates) {
        ++edge_processed;
        report_index_progress(config, vertex_progress_end, progress_end, edge_processed, edge_total, nullptr);
        if (edge.axis == 0u) {
            if (edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y - 1u, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y - 1u, edge.z)
            );
        } else if (edge.axis == 1u) {
            if (edge.y >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.z == 0u || edge.z >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z - 1u),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y, edge.z)
            );
        } else {
            if (edge.z >= cube_resolution || edge.x == 0u || edge.x >= cube_resolution || edge.y == 0u || edge.y >= cube_resolution) {
                continue;
            }
            append_surface_net_quad(
                surface_net,
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y - 1u, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y - 1u, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x, edge.y, edge.z),
                surface_net_cube_vertex(cube_vertices, cube_resolution, edge.x - 1u, edge.y, edge.z)
            );
        }
    }

    return surface_net;
}

void append_local_surface_net_patches(
    SurfaceNetMesh& surface_net,
    const QuantizedMesh& mesh,
    const std::vector<LocalSurfaceNetPatch>& patches,
    float grid_radius,
    const MarchingCubesConfig& config
) {
    if (patches.empty()) {
        return;
    }

    surface_net.local_patch_depth = patches.front().depth;
    uint64_t patch_index = 0;
    const uint64_t patch_total = patches.size();
    for (const LocalSurfaceNetPatch& patch : patches) {
        const double patch_begin = 0.86 + (0.90 - 0.86) * (static_cast<double>(patch_index) / static_cast<double>(patch_total));
        const double patch_end = 0.86 + (0.90 - 0.86) * (static_cast<double>(patch_index + 1u) / static_cast<double>(patch_total));
        ++patch_index;
        std::vector<LocalSurfaceNetPatch> root_patches;
        root_patches.push_back(patch);

        for (const LocalSurfaceNetPatch& root_patch : root_patches) {
            const uint32_t vertex_count_before = static_cast<uint32_t>(surface_net.vertices.size());
            const uint32_t triangle_count_before = static_cast<uint32_t>(surface_net.triangle_indices.size() / 3u);
            SurfaceNetMesh patch_mesh = build_local_surface_net_patch_mesh(mesh, root_patch, grid_radius, config, patch_begin, patch_end);
            if (!patch_mesh.vertices.empty() && !patch_mesh.triangle_indices.empty()) {
                const uint32_t patch_depth = std::clamp(root_patch.depth, 1u, MaxSvoDepth);
                const float depth_voxel_size = (grid_radius * 2.0f) / static_cast<float>(1u << patch_depth);
                if (root_patch.promoted_depth8_keys.empty()) {
                    suppress_surface_net_triangles_for_patches(
                        patch_mesh,
                        patches,
                        std::max(depth_voxel_size * 2.0f, kilometers_to_world_units(0.25f)),
                        false
                    );
                }
                if (root_patch.promoted_depth8_keys.empty()) {
                    bias_surface_net_outward(patch_mesh, depth_voxel_size * 0.25f);
                }
                if (!patch_mesh.triangle_indices.empty()) {
                    append_surface_net_mesh(surface_net, patch_mesh);
                }
            }

            const uint32_t added_vertices = static_cast<uint32_t>(surface_net.vertices.size()) - vertex_count_before;
            const uint32_t added_triangles = static_cast<uint32_t>(surface_net.triangle_indices.size() / 3u) - triangle_count_before;
            if (added_vertices == 0u || added_triangles == 0u) {
                continue;
            }
            ++surface_net.local_patch_count;
            surface_net.local_vertex_count += added_vertices;
            surface_net.local_triangle_count += added_triangles;
        }
    }
}

void build_svo_node_at(
    std::vector<SparseVoxelOctreeNode>& nodes,
    uint32_t node_index,
    const std::vector<VoxelKey>& keys,
    uint32_t begin,
    uint32_t end,
    uint32_t depth,
    uint32_t origin_x,
    uint32_t origin_y,
    uint32_t origin_z,
    uint32_t size,
    uint32_t max_depth
) {
    SparseVoxelOctreeNode& node = nodes[node_index];
    node.occupied_leaf_count = end - begin;
    node.depth = depth;
    node.origin_x = origin_x;
    node.origin_y = origin_y;
    node.origin_z = origin_z;
    node.size = size;

    if (begin >= end || depth >= max_depth || size <= 1u) {
        return;
    }

    const uint32_t child_size = size / 2u;
    std::array<uint32_t, 8> child_begin = {};
    std::array<uint32_t, 8> child_end = {};
    std::array<bool, 8> has_child = {};
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t child = child_index_for_voxel(keys[i], origin_x, origin_y, origin_z, child_size);
        if (!has_child[child]) {
            child_begin[child] = i;
            has_child[child] = true;
            node.child_mask |= (1u << child);
        }
        child_end[child] = i + 1u;
    }

    const uint32_t child_count = static_cast<uint32_t>(std::popcount(node.child_mask));
    node.first_child = static_cast<uint32_t>(nodes.size());
    const uint32_t first_child = node.first_child;
    nodes.resize(nodes.size() + child_count);

    uint32_t child_slot = 0;
    for (uint32_t child = 0; child < 8; ++child) {
        if (!has_child[child]) {
            continue;
        }
        build_svo_node_at(
            nodes,
            first_child + child_slot,
            keys,
            child_begin[child],
            child_end[child],
            depth + 1u,
            child_origin_x(origin_x, child, child_size),
            child_origin_y(origin_y, child, child_size),
            child_origin_z(origin_z, child, child_size),
            child_size,
            max_depth
        );
        ++child_slot;
    }
}

void count_svo_debug_boxes_recursive(
    const SparseVoxelOctree& svo,
    uint32_t node_index,
    uint32_t draw_depth,
    uint32_t max_boxes,
    uint32_t& count
) {
    if (node_index >= svo.nodes.size() || count >= max_boxes) {
        return;
    }

    const SparseVoxelOctreeNode& node = svo.nodes[node_index];
    if (node.child_mask == 0u || node.depth >= draw_depth) {
        ++count;
        return;
    }

    uint32_t child_slot = 0;
    for (uint32_t child = 0; child < 8; ++child) {
        if ((node.child_mask & (1u << child)) == 0u) {
            continue;
        }
        count_svo_debug_boxes_recursive(svo, node.first_child + child_slot, draw_depth, max_boxes, count);
        ++child_slot;
    }
}

std::vector<VoxelKey> build_base_voxel_occupancy(
    const QuantizedMesh& mesh,
    float grid_radius,
    float voxel_size,
    uint32_t resolution,
    const MarchingCubesConfig& config
) {
    std::vector<VoxelKey> keys;
    const uint32_t triangle_count = static_cast<uint32_t>((mesh.triangle_indices.size() + mesh.stitch_triangle_indices.size()) / 3u);
    keys.reserve(static_cast<size_t>(triangle_count) * 16u);
    const uint64_t total_sample_vertices =
        voxelized_triangle_sample_count(mesh, mesh.triangle_indices, voxel_size) +
        voxelized_triangle_sample_count(mesh, mesh.stitch_triangle_indices, voxel_size);
    uint64_t processed_sample_vertices = 0;
    report_progress(config, 0.56f, "Voxelizing terrain vertices");
    add_voxelized_triangles(
        keys,
        mesh,
        mesh.triangle_indices,
        grid_radius,
        voxel_size,
        resolution,
        config,
        0.56f,
        0.64f,
        processed_sample_vertices,
        total_sample_vertices
    );
    add_voxelized_triangles(
        keys,
        mesh,
        mesh.stitch_triangle_indices,
        grid_radius,
        voxel_size,
        resolution,
        config,
        0.56f,
        0.64f,
        processed_sample_vertices,
        total_sample_vertices
    );
    report_progress(config, 0.645f, "Sorting voxel occupancy");
    sort_and_unique_voxel_keys(keys);
    return keys;
}

float mesh_voxel_bounds_radius(const QuantizedMesh& mesh) {
    float max_radius = 1.0f;
    for (const QuantizedMeshVertex& vertex : mesh.vertices) {
        max_radius = std::max(max_radius, length(vertex.position));
    }
    return max_radius * 1.025f;
}

void generate_sparse_voxel_octree(QuantizedMesh& mesh, const MarchingCubesConfig& config) {
    const auto voxel_begin = PerfClock::now();
    validate_morton_helpers_once();

    mesh.svo = {};
    mesh.surface_net = {};
    if (!config.enable_svo_generation || mesh.vertices.empty()) {
        report_progress(config, 0.96f, "Voxel generation disabled");
        mesh.perf.voxel_total_ms = elapsed_ms(voxel_begin);
        return;
    }

    const VoxelEditSet promoted_terrain_edits = promoted_terrain_dig_edits(config);
    const bool should_build_exterior_replacement =
        config.enable_exterior_surface_net_replacement ||
        !promoted_terrain_edits.digs.empty() ||
        !promoted_terrain_edits.terrain_masks.empty();
    const bool promoted_terrain_only =
        !config.enable_exterior_surface_net_replacement &&
        (!promoted_terrain_edits.digs.empty() || !promoted_terrain_edits.terrain_masks.empty());

    if (config.voxel_features.enabled && !should_build_exterior_replacement) {
        report_progress(config, 0.97f, "Cave interiors queued for streaming");
        mesh.perf.cave_feature_count = static_cast<uint32_t>(mesh.voxel_features.size());
        mesh.surface_net = {};
        mesh.svo = {};
        mesh.perf.voxel_total_ms = elapsed_ms(voxel_begin);
        return;
    }

    if (!should_build_exterior_replacement) {
        report_progress(config, 0.97f, "Exterior SVO replacement disabled");
        mesh.surface_net = {};
        mesh.svo = {};
        mesh.perf.voxel_total_ms = elapsed_ms(voxel_begin);
        return;
    }

    MarchingCubesConfig exterior_config = config;
    if (!config.enable_exterior_surface_net_replacement) {
        exterior_config.voxel_edits = promoted_terrain_edits;
    }

    report_progress(config, 0.52f, "Preparing voxel grid");
    const uint32_t depth = std::clamp(exterior_config.svo_depth, 1u, MaxSvoDepth);
    const uint32_t resolution = 1u << depth;
    float grid_radius = mesh.voxel_occupancy_cache.depth == depth && mesh.voxel_occupancy_cache.bounds_radius > 0.0f
        ? mesh.voxel_occupancy_cache.bounds_radius
        : mesh_voxel_bounds_radius(mesh);
    const float voxel_size = (grid_radius * 2.0f) / static_cast<float>(resolution);

    if (mesh.voxel_occupancy_cache.depth != depth ||
        mesh.voxel_occupancy_cache.bounds_radius <= 0.0f ||
        mesh.voxel_occupancy_cache.leaf_keys.empty()) {
        report_progress(config, 0.56f, depth >= MaxSvoDepth ? "Voxelizing depth-16 terrain triangles" : "Voxelizing depth-8 global backbone");
        const auto voxelize_begin = PerfClock::now();
        mesh.voxel_occupancy_cache.leaf_keys = build_base_voxel_occupancy(mesh, grid_radius, voxel_size, resolution, exterior_config);
        mesh.perf.exterior_voxelize_ms = elapsed_ms(voxelize_begin);
        mesh.voxel_occupancy_cache.bounds_radius = grid_radius;
        mesh.voxel_occupancy_cache.depth = depth;
    } else {
        grid_radius = mesh.voxel_occupancy_cache.bounds_radius;
    }

    report_progress(config, 0.66f, "Preparing surface-net patches");
    const std::vector<LocalSurfaceNetPatch> local_surface_net_patches = realized_local_surface_net_patches(
        build_local_surface_net_patches(exterior_config, grid_radius),
        grid_radius
    );
    const bool has_promoted_depth8_roots = std::any_of(
        local_surface_net_patches.begin(),
        local_surface_net_patches.end(),
        [](const LocalSurfaceNetPatch& patch) {
            return patch.replace_surface && !patch.promoted_depth8_keys.empty();
        }
    );
    const uint32_t surface_net_depth = std::clamp(std::min(exterior_config.surface_net_depth, depth), 1u, MaxSvoDepth);
    if (mesh.surface_net_base_cache.source_depth != surface_net_depth ||
        std::fabs(mesh.surface_net_base_cache.bounds_radius - grid_radius) > 0.000001f ||
        mesh.surface_net_base_cache.material_id != exterior_config.surface_net_material_id ||
        mesh.surface_net_base_cache.vertices.empty()) {
        report_progress(config, 0.72f, "Building base surface net");
        const auto surface_begin = PerfClock::now();
        MarchingCubesConfig base_surface_config = exterior_config;
        base_surface_config.voxel_edits.digs.clear();
        const double base_surface_progress_end = has_promoted_depth8_roots ? 0.79 : 0.86;
        mesh.surface_net_base_cache = build_surface_net_mesh_from_occupancy(
            mesh.voxel_occupancy_cache.leaf_keys,
            depth,
            grid_radius,
            base_surface_config,
            {},
            0.72,
            base_surface_progress_end
        );
        mesh.perf.exterior_surface_net_ms += elapsed_ms(surface_begin);
    }
    if (has_promoted_depth8_roots) {
        report_progress(config, 0.80f, "Replacing edited surface roots");
        const auto replacement_surface_begin = PerfClock::now();
        if (promoted_terrain_only) {
            mesh.surface_net = {};
            mesh.surface_net.source_depth = surface_net_depth;
            mesh.surface_net.bounds_radius = grid_radius;
            mesh.surface_net.material_id = exterior_config.surface_net_material_id;
        } else {
            MarchingCubesConfig promoted_surface_config = exterior_config;
            promoted_surface_config.voxel_edits.digs.clear();
            mesh.surface_net = build_surface_net_mesh_from_occupancy(
                mesh.voxel_occupancy_cache.leaf_keys,
                depth,
                grid_radius,
                promoted_surface_config,
                local_surface_net_patches,
                0.80,
                0.85
            );
        }
        mesh.perf.exterior_surface_net_ms += elapsed_ms(replacement_surface_begin);
    } else {
        mesh.surface_net = mesh.surface_net_base_cache;
    }
    mesh.surface_net.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    mesh.surface_net.local_edit_depth = (config.voxel_edits.digs.empty() && config.voxel_edits.terrain_masks.empty()) ? 0u : config.voxel_edits.local_depth;
    if (has_promoted_depth8_roots) {
        report_progress(config, 0.86f, "Building local depth-16 patches");
        append_local_surface_net_patches(mesh.surface_net, mesh, local_surface_net_patches, grid_radius, exterior_config);
    }

    report_progress(config, 0.91f, "Applying voxel edits");
    std::vector<VoxelKey> keys = mesh.voxel_occupancy_cache.leaf_keys;
    const uint32_t dig_removed_leaf_count = apply_exact_voxel_dig_edits(keys, grid_radius, voxel_size, exterior_config.voxel_edits);

    mesh.svo.bounds_radius = grid_radius;
    mesh.svo.depth = depth;
    mesh.svo.max_depth = depth;
    mesh.svo.debug_draw_depth = std::min(config.svo_debug_draw_depth, depth);
    mesh.svo.debug_max_boxes = std::max(1u, config.svo_debug_max_boxes);
    mesh.svo.occupied_leaf_count = static_cast<uint32_t>(keys.size());
    mesh.svo.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    mesh.svo.dig_removed_leaf_count = dig_removed_leaf_count;
    mesh.svo.local_edit_depth = (config.voxel_edits.digs.empty() && config.voxel_edits.terrain_masks.empty()) ? 0u : config.voxel_edits.local_depth;
    if (!keys.empty()) {
        report_progress(config, 0.94f, "Building sparse voxel octree");
        const uint32_t debug_tree_depth = std::min(depth, mesh.svo.debug_draw_depth);
        mesh.svo.nodes.resize(1);
        build_svo_node_at(mesh.svo.nodes, 0, keys, 0, static_cast<uint32_t>(keys.size()), 0, 0, 0, 0, resolution, debug_tree_depth);
        count_svo_debug_boxes_recursive(mesh.svo, 0, debug_tree_depth, mesh.svo.debug_max_boxes, mesh.svo.debug_box_count);
    }
    mesh.perf.voxel_total_ms = elapsed_ms(voxel_begin);
    report_progress(config, 0.97f, "Voxel data ready");
}

} // namespace

SurfaceNetMesh build_cave_surface_net_for_feature(
    const LocalVoxelFeature& feature,
    MarchingCubesConfig config,
    float grid_radius,
    CaveFeatureBuildStats* stats
) {
    LocalVoxelFeature stream_feature = feature;
    stream_feature.svo_depth = std::clamp(config.voxel_features.cave_depth, 1u, MaxSvoDepth);
    config.progress_callback = {};
    std::vector<VoxelKey> sign_changing_keys;
    sign_changing_keys.reserve(24576u);
    return build_cave_surface_net_feature(
        stream_feature,
        config,
        grid_radius,
        sign_changing_keys,
        stats
    );
}

QuantizedMesh build_quantized_marching_cubes(
    const GoldbergTopology& topology,
    const PointCloud& points,
    const MarchingCubesConfig& config
) {
    const auto total_begin = PerfClock::now();
    QuantizedMesh mesh;
    report_progress(config, 0.02f, "Building Goldberg cell mesh");
    mesh.cell_count = static_cast<uint32_t>(topology.cells.size());
    BoundaryEdgeMap boundary_edges;
    boundary_edges.reserve(topology.cells.size() * 4096u);
    const FractureBuildCache fracture_cache = build_fracture_build_cache(config);
    const uint32_t plane_subdivisions = std::max(1u, std::max(config.resolution_x, config.resolution_y));
    mesh.min_cell_subdivisions = UINT32_MAX;
    mesh.max_cell_subdivisions = 0;
    mesh.lod_level_count = (config.enable_lod_subdivision_test || config.enable_camera_proximity_lod) ? std::max(1u, config.lod_levels) : 1u;
    std::vector<FractureSeed> fracture_seed_scratch;
    std::vector<FractureShard> fracture_shard_scratch;
    mesh.cave_anchor_points = build_cave_anchor_points(config);
    mesh.voxel_features = build_local_voxel_features(topology, config, mesh.cave_anchor_points);
    const std::vector<LocalVoxelFeature> baked_cave_features;

    const auto goldberg_begin = PerfClock::now();
    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        const float surface_radius = owned_surface_radius(points, cell_id);
        const uint32_t material_id = cell.kind == GoldbergCellKind::Pentagon ? 1u : 0u;
        const uint32_t cell_subdivisions = cell_lod_subdivisions(cell, config, plane_subdivisions);
        mesh.min_cell_subdivisions = std::min(mesh.min_cell_subdivisions, cell_subdivisions);
        mesh.max_cell_subdivisions = std::max(mesh.max_cell_subdivisions, cell_subdivisions);
        const auto cell_emit_begin = PerfClock::now();
        emit_subdivided_goldberg_cell_plane(
            mesh,
            boundary_edges,
            topology,
            cell_id,
            cell_subdivisions,
            surface_radius,
            material_id,
            config,
            fracture_cache,
            fracture_seed_scratch,
            fracture_shard_scratch,
            baked_cave_features
        );
        if (config.enable_fractures) {
            mesh.perf.fracture_mesh_ms += elapsed_ms(cell_emit_begin);
        }
        if ((cell_id % 8u) == 0u || cell_id + 1u == topology.cells.size()) {
            const float cell_progress = topology.cells.empty()
                ? 0.42f
                : static_cast<float>(cell_id + 1u) / static_cast<float>(topology.cells.size());
            report_progress(config, 0.04f + cell_progress * 0.38f, "Building Goldberg cell mesh");
        }
    }
    mesh.perf.goldberg_mesh_ms = elapsed_ms(goldberg_begin);

    if (topology.cells.empty()) {
        mesh.min_cell_subdivisions = 0;
    }
    report_progress(config, 0.44f, "Sorting boundary edges");
    const auto boundary_sort_begin = PerfClock::now();
    sort_and_merge_boundary_edges(boundary_edges);
    mesh.perf.boundary_sort_ms = elapsed_ms(boundary_sort_begin);
    report_progress(config, 0.48f, "Stitching LOD boundaries");
    const auto stitch_begin = PerfClock::now();
    build_transition_stitches(mesh, topology, points, boundary_edges, config, fracture_cache);
    mesh.perf.stitch_ms = elapsed_ms(stitch_begin);
    generate_sparse_voxel_octree(mesh, config);

    for (uint32_t i = 0; i + 2u < mesh.triangle_indices.size(); i += 3u) {
        const uint32_t index = mesh.triangle_indices[i];
        if (index < mesh.vertices.size()) {
            const uint32_t material = std::min(mesh.vertices[index].material_id, 7u);
            ++mesh.perf.material_triangle_counts[material];
        }
    }
    for (uint32_t i = 0; i + 2u < mesh.stitch_triangle_indices.size(); i += 3u) {
        const uint32_t index = mesh.stitch_triangle_indices[i];
        if (index < mesh.vertices.size()) {
            const uint32_t material = std::min(mesh.vertices[index].material_id, 7u);
            ++mesh.perf.material_triangle_counts[material];
        }
    }
    mesh.perf.total_ms = elapsed_ms(total_begin);

    return mesh;
}

QuantizedMesh rebuild_quantized_mesh_voxels(
    QuantizedMesh mesh,
    const MarchingCubesConfig& config
) {
    const auto total_begin = PerfClock::now();
    mesh.perf.voxel_total_ms = 0.0;
    mesh.perf.cave_surface_net_ms = 0.0;
    mesh.perf.cave_svo_ms = 0.0;
    mesh.perf.exterior_voxelize_ms = 0.0;
    mesh.perf.exterior_surface_net_ms = 0.0;
    mesh.perf.cave_features.clear();
    generate_sparse_voxel_octree(mesh, config);
    mesh.perf.total_ms = elapsed_ms(total_begin);
    return mesh;
}

QuantizedMeshValidation validate_quantized_mesh(const QuantizedMesh& mesh) {
    std::ostringstream report;
    bool ok = true;

    auto require = [&](bool condition, const char* message) {
        if (!condition) {
            ok = false;
            report << "Quantized MC failed: " << message << '\n';
        }
    };

    require(mesh.cell_count > 0, "cell count is zero");
    require(!mesh.vertices.empty(), "vertex buffer is empty");
    require(!mesh.triangle_indices.empty(), "triangle index buffer is empty");
    require(mesh.triangle_indices.size() % 3 == 0, "triangle index count is not divisible by 3");
    require(mesh.line_indices.size() % 2 == 0, "line index count is not divisible by 2");
    require(mesh.stitch_triangle_indices.size() % 3 == 0, "stitch triangle index count is not divisible by 3");
    require(mesh.stitch_line_indices.size() % 2 == 0, "stitch line index count is not divisible by 2");
    require(mesh.triangle_count == mesh.triangle_indices.size() / 3, "triangle count does not match index buffer");
    require(mesh.stitch_triangle_count == mesh.stitch_triangle_indices.size() / 3, "stitch triangle count does not match index buffer");
    require(mesh.stitch_triangle_count > 0, "stitch triangle count is zero");
    require(mesh.boundary_edge_count > 0, "boundary edge count is zero");
    require(mesh.boundary_run_count > 0, "boundary run count is zero");
    require(mesh.paired_boundary_run_count > 0, "paired boundary run count is zero");
    require(mesh.chain_stitch_triangle_count > 0, "chain stitch triangle count is zero");
    require(mesh.shared_edge_path_count > 0, "shared edge path count is zero");
    require(mesh.greedy_path_step_count > 0, "greedy path step count is zero");
    require(mesh.chain_stitch_triangle_count + mesh.fallback_stitch_triangle_count == mesh.stitch_triangle_count, "stitch source counts do not match stitch triangle count");
    require(mesh.fallback_stitch_triangle_count == 0, "fallback stitch triangle count is nonzero");
    require(mesh.min_cell_subdivisions > 0, "minimum cell subdivision count is zero");
    require(mesh.max_cell_subdivisions >= mesh.min_cell_subdivisions, "cell subdivision range is invalid");
    if (mesh.surface_net.source_depth > 0u) {
        require(!mesh.surface_net.vertices.empty(), "surface net vertex buffer is empty");
        require(mesh.surface_net.normals.size() == mesh.surface_net.vertices.size(), "surface net normal count does not match vertex count");
        require(mesh.surface_net.vertex_depths.empty() || mesh.surface_net.vertex_depths.size() == mesh.surface_net.vertices.size(), "surface net depth tag count does not match vertex count");
        require(!mesh.surface_net.triangle_indices.empty(), "surface net index buffer is empty");
        require(mesh.surface_net.triangle_indices.size() % 3 == 0, "surface net index count is not divisible by 3");
    }

    for (uint32_t index : mesh.triangle_indices) {
        require(index < mesh.vertices.size(), "triangle index out of range");
    }
    for (uint32_t index : mesh.line_indices) {
        require(index < mesh.vertices.size(), "line index out of range");
    }
    for (uint32_t index : mesh.stitch_triangle_indices) {
        require(index < mesh.vertices.size(), "stitch triangle index out of range");
    }
    for (uint32_t index : mesh.stitch_line_indices) {
        require(index < mesh.vertices.size(), "stitch line index out of range");
    }
    for (uint32_t index : mesh.surface_net.triangle_indices) {
        require(index < mesh.surface_net.vertices.size(), "surface net index out of range");
    }

    if (ok) {
        report << "Goldberg plane mesh OK: " << mesh.cell_count << " cells, "
               << mesh.vertices.size() << " vertices, "
               << mesh.min_cell_subdivisions << '-' << mesh.max_cell_subdivisions
               << " subdivisions/cell across " << mesh.lod_level_count << " LOD levels, "
               << mesh.triangle_count << " emitted triangles, "
               << mesh.rejected_triangle_count << " rejected triangles, "
               << mesh.clipped_triangle_count << " clipped triangles, "
               << mesh.discarded_clipped_triangle_count << " discarded clipped triangles, "
               << mesh.boundary_edge_count << " boundary edges, "
               << mesh.boundary_run_count << " boundary runs, "
               << mesh.paired_boundary_run_count << " paired runs, "
               << mesh.shared_edge_path_count << " shared-edge paths, "
               << mesh.stitch_triangle_count << " stitch triangles ("
               << mesh.chain_stitch_triangle_count << " chain, "
               << mesh.fallback_stitch_triangle_count << " fallback), "
               << mesh.greedy_path_step_count << " greedy path steps, "
               << mesh.rejected_greedy_jump_count << " rejected greedy jumps, "
               << mesh.rejected_stitch_run_count << " rejected stitch runs, "
               << mesh.unstitched_gap_count << " unstitched gaps, SVO depth "
               << mesh.svo.depth << ", "
               << mesh.svo.debug_draw_depth << " debug draw depth, "
               << mesh.svo.occupied_leaf_count << " occupied leaves, "
               << mesh.svo.nodes.size() << " SVO nodes, "
               << mesh.svo.debug_box_count << " debug boxes";
        if (mesh.svo.dig_edit_count > 0u) {
            report << ", " << mesh.svo.dig_edit_count << " dig edits, "
                   << mesh.svo.dig_removed_leaf_count << " carved leaves, local edit depth "
                   << mesh.svo.local_edit_depth;
        }
        if (mesh.svo.depth > 0 && mesh.svo.bounds_radius > 0.0f) {
            const float leaf_world_units = (mesh.svo.bounds_radius * 2.0f) / static_cast<float>(1u << mesh.svo.depth);
            report << ", " << world_units_to_kilometers(leaf_world_units) << " km SVO leaf size";
        }
        if (mesh.surface_net.source_depth > 0u) {
            report << ", surface net depth " << mesh.surface_net.source_depth
                   << ", " << mesh.surface_net.vertices.size() << " surface net vertices, "
                   << (mesh.surface_net.triangle_indices.size() / 3u) << " surface net triangles";
            if (mesh.surface_net.local_patch_count > 0u) {
                report << ", " << mesh.surface_net.local_patch_count << " local surface patches at depth "
                       << mesh.surface_net.local_patch_depth << " ("
                       << mesh.surface_net.local_vertex_count << " vertices, "
                       << mesh.surface_net.local_triangle_count << " triangles)";
            }
            if (mesh.surface_net.dig_edit_count > 0u) {
                report << ", surface net digs " << mesh.surface_net.dig_edit_count;
            }
        }
        report << ", perf total " << mesh.perf.total_ms << " ms"
               << " (Goldberg " << mesh.perf.goldberg_mesh_ms << " ms"
               << ", fractures " << mesh.perf.fracture_mesh_ms << " ms"
               << ", boundary sort " << mesh.perf.boundary_sort_ms << " ms"
               << ", stitch " << mesh.perf.stitch_ms << " ms"
               << ", voxels " << mesh.perf.voxel_total_ms << " ms"
               << ", cave nets " << mesh.perf.cave_surface_net_ms << " ms"
               << ", cave SVO " << mesh.perf.cave_svo_ms << " ms)";
        if (mesh.perf.cave_feature_count > 0u) {
            report << ", cave features " << mesh.perf.cave_feature_count
                   << " (" << mesh.perf.cave_surface_vertices << " vertices, "
                   << mesh.perf.cave_surface_triangles << " triangles)";
        }
        report << '.';
    }

    return {ok, report.str()};
}

} // namespace ae
