#include "aetheronus/meshing.hpp"
#include "aetheronus/marching_cubes_tables.hpp"
#include "aetheronus/planet_scale.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <vector>
#include <sstream>

namespace ae {
namespace {

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

struct VoxelKey {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
};

struct SurfaceNetEdgeKey {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t axis = 0;
};

struct PreparedVoxelDig {
    Vec3 center_mesh;
    float radius_with_leaf_mesh = 0.0f;
};

struct LocalSurfaceNetPatch {
    Vec3 center_mesh;
    float suppress_radius_mesh = 0.0f;
    float extraction_radius_mesh = 0.0f;
    uint32_t depth = 13;
};

using BoundaryEdgeMap = std::vector<BoundaryEdgeRecord>;
using BoundaryPairMap = std::map<BoundaryPairKey, BoundaryPairChains>;
using BoundaryPairRunMap = std::map<BoundaryPairKey, BoundaryPairRuns>;

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

uint64_t voxel_morton_code(VoxelKey key) {
    uint64_t code = 0;
    for (uint32_t bit = 0; bit < 21; ++bit) {
        code |= (static_cast<uint64_t>((key.z >> bit) & 1u) << (bit * 3u));
        code |= (static_cast<uint64_t>((key.y >> bit) & 1u) << (bit * 3u + 1u));
        code |= (static_cast<uint64_t>((key.x >> bit) & 1u) << (bit * 3u + 2u));
    }
    return code;
}

bool operator<(const VoxelKey& lhs, const VoxelKey& rhs) {
    const uint64_t lhs_code = voxel_morton_code(lhs);
    const uint64_t rhs_code = voxel_morton_code(rhs);
    if (lhs_code != rhs_code) {
        return lhs_code < rhs_code;
    }
    if (lhs.x != rhs.x) return lhs.x < rhs.x;
    if (lhs.y != rhs.y) return lhs.y < rhs.y;
    return lhs.z < rhs.z;
}

bool operator==(const VoxelKey& lhs, const VoxelKey& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
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
    std::vector<FractureShard>& fracture_shard_scratch
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
                    for (const FractureShard& shard : shards) {
                            emit_fractured_local_triangle(mesh, boundary_edges, cell_id, tri0, shard, cell.center, cell.normal, frame, cell_polygon, surface_radius, config, cache, material_id);
                    }
                } else {
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
                        for (const FractureShard& shard : shards) {
                            emit_fractured_local_triangle(mesh, boundary_edges, cell_id, tri1, shard, cell.center, cell.normal, frame, cell_polygon, surface_radius, config, cache, material_id);
                        }
                    } else {
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
        if (dig.radius_km <= 0.0f) {
            continue;
        }
        const float radius_mesh = kilometers_to_world_units(dig.radius_km);
        prepared.push_back({
            dig.center_mesh,
            radius_mesh + leaf_radius,
        });
    }
    return prepared;
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

void add_voxelized_triangles(
    std::vector<VoxelKey>& keys,
    const QuantizedMesh& mesh,
    const std::vector<uint32_t>& indices,
    float grid_radius,
    float voxel_size,
    uint32_t resolution
) {
    for (uint32_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vec3 a = mesh.vertices[indices[i]].position;
        const Vec3 b = mesh.vertices[indices[i + 1]].position;
        const Vec3 c = mesh.vertices[indices[i + 2]].position;
        const float longest_edge = std::max(length(b - a), std::max(length(c - b), length(a - c)));
        const float sample_spacing = std::max(voxel_size * 2.0f, 0.0001f);
        const uint32_t steps = std::clamp(static_cast<uint32_t>(std::ceil(longest_edge / sample_spacing)), 1u, 10u);
        for (uint32_t u_step = 0; u_step <= steps; ++u_step) {
            for (uint32_t v_step = 0; v_step + u_step <= steps; ++v_step) {
                const float u = static_cast<float>(u_step) / static_cast<float>(steps);
                const float v = static_cast<float>(v_step) / static_cast<float>(steps);
                const float w = 1.0f - u - v;
                keys.push_back(voxel_key_for_position(a * u + b * v + c * w, grid_radius, voxel_size, resolution));
            }
        }
        keys.push_back(voxel_key_for_position((a + b + c) / 3.0f, grid_radius, voxel_size, resolution));
        keys.push_back(voxel_key_for_position((a + b) * 0.5f, grid_radius, voxel_size, resolution));
        keys.push_back(voxel_key_for_position((b + c) * 0.5f, grid_radius, voxel_size, resolution));
        keys.push_back(voxel_key_for_position((c + a) * 0.5f, grid_radius, voxel_size, resolution));
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
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<LocalSurfaceNetPatch> build_local_surface_net_patches(const MarchingCubesConfig& config) {
    std::vector<LocalSurfaceNetPatch> patches;
    if (!config.enable_local_surface_net_detail || config.local_surface_net_max_patches == 0u) {
        return patches;
    }

    const float suppress_radius = kilometers_to_world_units(std::max(0.0f, config.local_surface_net_patch_radius_km));
    const float overlap = kilometers_to_world_units(std::max(0.0f, config.local_surface_net_patch_overlap_km));
    if (suppress_radius <= 0.0f) {
        return patches;
    }

    const float extraction_radius = suppress_radius + overlap;
    const uint32_t depth = std::clamp(config.local_surface_net_depth, 1u, 13u);
    auto add_patch = [&](Vec3 center) {
        if (patches.size() >= config.local_surface_net_max_patches || length(center) <= 0.000001f) {
            return;
        }
        center = normalize(center);
        for (const LocalSurfaceNetPatch& patch : patches) {
            if (length(patch.center_mesh - center) < suppress_radius * 0.5f) {
                return;
            }
        }
        patches.push_back({center, suppress_radius, extraction_radius, depth});
    };

    add_patch(config.lod_camera_position);

    struct DigCandidate {
        Vec3 center;
        float distance_to_camera = 0.0f;
        uint32_t newest_rank = 0;
    };

    const Vec3 camera_center = length(config.lod_camera_position) > 0.000001f
        ? normalize(config.lod_camera_position)
        : Vec3{0.0f, 0.0f, 1.0f};
    std::vector<DigCandidate> dig_candidates;
    dig_candidates.reserve(config.voxel_edits.digs.size());
    for (uint32_t i = 0; i < config.voxel_edits.digs.size(); ++i) {
        const VoxelDigEdit& dig = config.voxel_edits.digs[i];
        if (length(dig.center_mesh) <= 0.000001f) {
            continue;
        }
        dig_candidates.push_back({
            normalize(dig.center_mesh),
            length(normalize(dig.center_mesh) - camera_center),
            static_cast<uint32_t>(config.voxel_edits.digs.size() - i),
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
        add_patch(dig.center);
    }

    return patches;
}

bool surface_net_key_in_patch(Vec3 center, float leaf_radius, const std::vector<LocalSurfaceNetPatch>& patches) {
    for (const LocalSurfaceNetPatch& patch : patches) {
        if (length(center - patch.center_mesh) <= patch.suppress_radius_mesh + leaf_radius) {
            return true;
        }
    }
    return false;
}

void add_surface_net_edge_candidate(
    std::vector<SurfaceNetEdgeKey>& edges,
    const std::vector<uint64_t>& occupancy,
    uint32_t resolution,
    uint32_t x,
    uint32_t y,
    uint32_t z,
    uint32_t axis,
    bool negative_direction
) {
    if (axis == 0u) {
        if (negative_direction) {
            if (x == 0u) return;
            const bool neighbor = surface_net_bit(occupancy, surface_net_grid_index(x - 1u, y, z, resolution));
            if (!neighbor) edges.push_back({x - 1u, y, z, axis});
        } else if (x + 1u < resolution) {
            const bool neighbor = surface_net_bit(occupancy, surface_net_grid_index(x + 1u, y, z, resolution));
            if (!neighbor) edges.push_back({x, y, z, axis});
        }
    } else if (axis == 1u) {
        if (negative_direction) {
            if (y == 0u) return;
            const bool neighbor = surface_net_bit(occupancy, surface_net_grid_index(x, y - 1u, z, resolution));
            if (!neighbor) edges.push_back({x, y - 1u, z, axis});
        } else if (y + 1u < resolution) {
            const bool neighbor = surface_net_bit(occupancy, surface_net_grid_index(x, y + 1u, z, resolution));
            if (!neighbor) edges.push_back({x, y, z, axis});
        }
    } else {
        if (negative_direction) {
            if (z == 0u) return;
            const bool neighbor = surface_net_bit(occupancy, surface_net_grid_index(x, y, z - 1u, resolution));
            if (!neighbor) edges.push_back({x, y, z - 1u, axis});
        } else if (z + 1u < resolution) {
            const bool neighbor = surface_net_bit(occupancy, surface_net_grid_index(x, y, z + 1u, resolution));
            if (!neighbor) edges.push_back({x, y, z, axis});
        }
    }
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

SurfaceNetMesh build_surface_net_mesh_from_occupancy(
    const std::vector<VoxelKey>& sorted_occupied_keys,
    uint32_t source_depth,
    float grid_radius,
    const MarchingCubesConfig& config,
    const std::vector<LocalSurfaceNetPatch>& suppression_patches = {}
) {
    SurfaceNetMesh surface_net;
    if (!config.enable_surface_net_generation || sorted_occupied_keys.empty()) {
        return surface_net;
    }

    const uint32_t target_depth = std::clamp(std::min(config.surface_net_depth, source_depth), 1u, 8u);
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
                    return surface_net_key_in_patch(
                        voxel_center_from_key(key, grid_radius, voxel_size),
                        voxel_radius,
                        suppression_patches
                    );
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

    std::vector<VoxelKey> cube_candidates;
    cube_candidates.reserve(occupied_keys.size() * 8u);
    std::vector<SurfaceNetEdgeKey> edge_candidates;
    edge_candidates.reserve(occupied_keys.size() * 6u);
    for (VoxelKey key : occupied_keys) {
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
                    if (cube_x < cube_resolution && cube_y < cube_resolution && cube_z < cube_resolution) {
                        cube_candidates.push_back({cube_x, cube_y, cube_z});
                    }
                }
            }
        }

        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 0u, false);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 0u, true);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 1u, false);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 1u, true);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 2u, false);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 2u, true);
    }

    std::sort(cube_candidates.begin(), cube_candidates.end());
    cube_candidates.erase(std::unique(cube_candidates.begin(), cube_candidates.end()), cube_candidates.end());
    std::sort(edge_candidates.begin(), edge_candidates.end());
    edge_candidates.erase(std::unique(edge_candidates.begin(), edge_candidates.end()), edge_candidates.end());

    const uint32_t cube_count = cube_resolution * cube_resolution * cube_resolution;
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

    constexpr std::array<std::array<uint32_t, 3>, 8> corners = {{
        {{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{1, 1, 0}},
        {{0, 0, 1}}, {{1, 0, 1}}, {{0, 1, 1}}, {{1, 1, 1}},
    }};
    constexpr std::array<std::array<uint32_t, 2>, 12> edges = {{
        {{0, 1}}, {{0, 2}}, {{1, 3}}, {{2, 3}},
        {{4, 5}}, {{4, 6}}, {{5, 7}}, {{6, 7}},
        {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
    }};

    for (VoxelKey cube : cube_candidates) {
        if (surface_net.vertices.size() >= max_vertices) {
            break;
        }

        std::array<bool, 8> inside = {};
        std::array<Vec3, 8> positions = {};
        uint32_t inside_count = 0;
        for (uint32_t i = 0; i < corners.size(); ++i) {
            const uint32_t x = cube.x + corners[i][0];
            const uint32_t y = cube.y + corners[i][1];
            const uint32_t z = cube.z + corners[i][2];
            inside[i] = surface_net_bit(occupancy, surface_net_grid_index(x, y, z, resolution));
            inside_count += inside[i] ? 1u : 0u;
            positions[i] = surface_net_sample_position(x, y, z, grid_radius, voxel_size);
        }
        if (inside_count == 0u || inside_count == 8u) {
            continue;
        }

        Vec3 average = {};
        Vec3 normal = {};
        uint32_t crossing_count = 0;
        for (const auto& edge : edges) {
            const uint32_t a = edge[0];
            const uint32_t b = edge[1];
            if (inside[a] == inside[b]) {
                continue;
            }
            average = average + (positions[a] + positions[b]) * 0.5f;
            normal = normal + (inside[a] ? positions[b] - positions[a] : positions[a] - positions[b]);
            ++crossing_count;
        }
        if (crossing_count == 0u) {
            continue;
        }

        const uint32_t vertex_index = static_cast<uint32_t>(surface_net.vertices.size());
        surface_net.vertices.push_back(average / static_cast<float>(crossing_count));
        surface_net.normals.push_back(length(normal) > 0.000001f ? normalize(normal) : normalize(average));
        cube_vertices[surface_net_cube_index(cube.x, cube.y, cube.z, cube_resolution)] = static_cast<int32_t>(vertex_index);
    }

    surface_net.triangle_indices.reserve(edge_candidates.size() * 6u);
    for (SurfaceNetEdgeKey edge : edge_candidates) {
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
    destination.triangle_indices.reserve(destination.triangle_indices.size() + source.triangle_indices.size());
    for (uint32_t index : source.triangle_indices) {
        destination.triangle_indices.push_back(vertex_offset + index);
    }
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

bool local_surface_net_sample_carved(Vec3 position, const std::vector<PreparedVoxelDig>& digs) {
    for (const PreparedVoxelDig& dig : digs) {
        if (length(position - dig.center_mesh) <= dig.radius_with_leaf_mesh) {
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
    const std::vector<PreparedVoxelDig>& digs
) {
    if (length(position - patch.center_mesh) > patch.extraction_radius_mesh || local_surface_net_sample_carved(position, digs)) {
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
    const std::vector<PreparedVoxelDig>& digs
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
                push_local_voxel_key(keys, a * u + b * v + c * w, patch, origin, voxel_size, resolution, digs);
            }
        }
        push_local_voxel_key(keys, (a + b + c) / 3.0f, patch, origin, voxel_size, resolution, digs);
        push_local_voxel_key(keys, (a + b) * 0.5f, patch, origin, voxel_size, resolution, digs);
        push_local_voxel_key(keys, (b + c) * 0.5f, patch, origin, voxel_size, resolution, digs);
        push_local_voxel_key(keys, (c + a) * 0.5f, patch, origin, voxel_size, resolution, digs);
    }
}

SurfaceNetMesh build_local_surface_net_patch_mesh(
    const QuantizedMesh& mesh,
    const LocalSurfaceNetPatch& patch,
    float grid_radius,
    const MarchingCubesConfig& config
) {
    SurfaceNetMesh surface_net;
    if (!config.enable_surface_net_generation) {
        return surface_net;
    }

    const uint32_t depth = std::clamp(patch.depth, 1u, 13u);
    const uint32_t global_resolution = 1u << depth;
    const float voxel_size = (grid_radius * 2.0f) / static_cast<float>(global_resolution);
    uint32_t resolution = static_cast<uint32_t>(std::ceil((patch.extraction_radius_mesh * 2.0f) / voxel_size)) + 4u;
    resolution = std::clamp(resolution, 4u, 256u);
    const Vec3 origin = patch.center_mesh - Vec3{
        voxel_size * static_cast<float>(resolution) * 0.5f,
        voxel_size * static_cast<float>(resolution) * 0.5f,
        voxel_size * static_cast<float>(resolution) * 0.5f,
    };

    std::vector<VoxelKey> occupied_keys;
    occupied_keys.reserve(32768);
    const std::vector<PreparedVoxelDig> digs = prepare_voxel_digs(config.voxel_edits, voxel_size);
    add_local_voxelized_triangles(occupied_keys, mesh, mesh.triangle_indices, patch, origin, voxel_size, resolution, digs);
    add_local_voxelized_triangles(occupied_keys, mesh, mesh.stitch_triangle_indices, patch, origin, voxel_size, resolution, digs);
    std::sort(occupied_keys.begin(), occupied_keys.end());
    occupied_keys.erase(std::unique(occupied_keys.begin(), occupied_keys.end()), occupied_keys.end());
    if (occupied_keys.empty()) {
        return surface_net;
    }

    const uint32_t cube_resolution = resolution - 1u;
    const uint32_t grid_count = resolution * resolution * resolution;
    std::vector<uint64_t> occupancy((grid_count + 63u) / 64u, 0u);
    for (VoxelKey key : occupied_keys) {
        if (key.x < resolution && key.y < resolution && key.z < resolution) {
            set_surface_net_bit(occupancy, surface_net_grid_index(key.x, key.y, key.z, resolution));
        }
    }

    std::vector<VoxelKey> cube_candidates;
    cube_candidates.reserve(occupied_keys.size() * 8u);
    std::vector<SurfaceNetEdgeKey> edge_candidates;
    edge_candidates.reserve(occupied_keys.size() * 6u);
    for (VoxelKey key : occupied_keys) {
        for (uint32_t dz = 0; dz < 2; ++dz) {
            for (uint32_t dy = 0; dy < 2; ++dy) {
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    if ((dx == 1u && key.x == 0u) || (dy == 1u && key.y == 0u) || (dz == 1u && key.z == 0u)) {
                        continue;
                    }
                    const uint32_t cube_x = key.x - dx;
                    const uint32_t cube_y = key.y - dy;
                    const uint32_t cube_z = key.z - dz;
                    if (cube_x < cube_resolution && cube_y < cube_resolution && cube_z < cube_resolution) {
                        cube_candidates.push_back({cube_x, cube_y, cube_z});
                    }
                }
            }
        }

        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 0u, false);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 0u, true);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 1u, false);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 1u, true);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 2u, false);
        add_surface_net_edge_candidate(edge_candidates, occupancy, resolution, key.x, key.y, key.z, 2u, true);
    }

    std::sort(cube_candidates.begin(), cube_candidates.end());
    cube_candidates.erase(std::unique(cube_candidates.begin(), cube_candidates.end()), cube_candidates.end());
    std::sort(edge_candidates.begin(), edge_candidates.end());
    edge_candidates.erase(std::unique(edge_candidates.begin(), edge_candidates.end()), edge_candidates.end());

    const uint32_t max_vertices = std::max(1u, config.surface_net_max_vertices);
    const uint32_t cube_count = cube_resolution * cube_resolution * cube_resolution;
    std::vector<int32_t> cube_vertices(cube_count, -1);
    surface_net.source_depth = depth;
    surface_net.bounds_radius = grid_radius;
    surface_net.occupied_voxel_count = static_cast<uint32_t>(occupied_keys.size());
    surface_net.candidate_cube_count = static_cast<uint32_t>(cube_candidates.size());
    surface_net.material_id = config.surface_net_material_id;
    surface_net.vertices.reserve(std::min<uint32_t>(surface_net.candidate_cube_count, max_vertices));
    surface_net.normals.reserve(surface_net.vertices.capacity());

    constexpr std::array<std::array<uint32_t, 3>, 8> corners = {{
        {{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{1, 1, 0}},
        {{0, 0, 1}}, {{1, 0, 1}}, {{0, 1, 1}}, {{1, 1, 1}},
    }};
    constexpr std::array<std::array<uint32_t, 2>, 12> edges = {{
        {{0, 1}}, {{0, 2}}, {{1, 3}}, {{2, 3}},
        {{4, 5}}, {{4, 6}}, {{5, 7}}, {{6, 7}},
        {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
    }};

    for (VoxelKey cube : cube_candidates) {
        if (surface_net.vertices.size() >= max_vertices) {
            break;
        }

        std::array<bool, 8> inside = {};
        std::array<Vec3, 8> positions = {};
        uint32_t inside_count = 0;
        for (uint32_t i = 0; i < corners.size(); ++i) {
            const uint32_t x = cube.x + corners[i][0];
            const uint32_t y = cube.y + corners[i][1];
            const uint32_t z = cube.z + corners[i][2];
            inside[i] = surface_net_bit(occupancy, surface_net_grid_index(x, y, z, resolution));
            inside_count += inside[i] ? 1u : 0u;
            positions[i] = local_surface_net_sample_position(x, y, z, origin, voxel_size);
        }
        if (inside_count == 0u || inside_count == 8u) {
            continue;
        }

        Vec3 average = {};
        Vec3 normal = {};
        uint32_t crossing_count = 0;
        for (const auto& edge : edges) {
            const uint32_t a = edge[0];
            const uint32_t b = edge[1];
            if (inside[a] == inside[b]) {
                continue;
            }
            average = average + (positions[a] + positions[b]) * 0.5f;
            normal = normal + (inside[a] ? positions[b] - positions[a] : positions[a] - positions[b]);
            ++crossing_count;
        }
        if (crossing_count == 0u) {
            continue;
        }

        const Vec3 vertex = average / static_cast<float>(crossing_count);
        if (length(vertex - patch.center_mesh) > patch.extraction_radius_mesh) {
            continue;
        }
        const uint32_t vertex_index = static_cast<uint32_t>(surface_net.vertices.size());
        surface_net.vertices.push_back(vertex);
        surface_net.normals.push_back(length(normal) > 0.000001f ? normalize(normal) : normalize(vertex));
        cube_vertices[surface_net_cube_index(cube.x, cube.y, cube.z, cube_resolution)] = static_cast<int32_t>(vertex_index);
    }

    surface_net.triangle_indices.reserve(edge_candidates.size() * 6u);
    for (SurfaceNetEdgeKey edge : edge_candidates) {
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
    for (const LocalSurfaceNetPatch& patch : patches) {
        const uint32_t vertex_count_before = static_cast<uint32_t>(surface_net.vertices.size());
        const uint32_t triangle_count_before = static_cast<uint32_t>(surface_net.triangle_indices.size() / 3u);
        SurfaceNetMesh patch_mesh = build_local_surface_net_patch_mesh(mesh, patch, grid_radius, config);
        if (patch_mesh.vertices.empty() || patch_mesh.triangle_indices.empty()) {
            continue;
        }
        append_surface_net_mesh(surface_net, patch_mesh);
        ++surface_net.local_patch_count;
        surface_net.local_vertex_count += static_cast<uint32_t>(surface_net.vertices.size()) - vertex_count_before;
        surface_net.local_triangle_count += static_cast<uint32_t>(surface_net.triangle_indices.size() / 3u) - triangle_count_before;
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

void generate_sparse_voxel_octree(QuantizedMesh& mesh, const MarchingCubesConfig& config) {
    mesh.svo = {};
    mesh.surface_net = {};
    if (!config.enable_svo_generation || mesh.vertices.empty()) {
        return;
    }

    const uint32_t depth = std::clamp(config.svo_depth, 1u, 13u);
    const uint32_t resolution = 1u << depth;
    float max_radius = 1.0f;
    for (const QuantizedMeshVertex& vertex : mesh.vertices) {
        max_radius = std::max(max_radius, length(vertex.position));
    }

    const float grid_radius = max_radius * 1.025f;
    const float voxel_size = (grid_radius * 2.0f) / static_cast<float>(resolution);
    std::vector<VoxelKey> keys;
    const uint32_t triangle_count = static_cast<uint32_t>((mesh.triangle_indices.size() + mesh.stitch_triangle_indices.size()) / 3u);
    keys.reserve(static_cast<size_t>(triangle_count) * 16u);
    add_voxelized_triangles(keys, mesh, mesh.triangle_indices, grid_radius, voxel_size, resolution);
    add_voxelized_triangles(keys, mesh, mesh.stitch_triangle_indices, grid_radius, voxel_size, resolution);

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const uint32_t dig_removed_leaf_count = apply_voxel_dig_edits(keys, grid_radius, voxel_size, config.voxel_edits);
    const std::vector<LocalSurfaceNetPatch> local_surface_net_patches = build_local_surface_net_patches(config);
    mesh.surface_net = build_surface_net_mesh_from_occupancy(keys, depth, grid_radius, config, local_surface_net_patches);
    append_local_surface_net_patches(mesh.surface_net, mesh, local_surface_net_patches, grid_radius, config);

    mesh.svo.bounds_radius = grid_radius;
    mesh.svo.depth = depth;
    mesh.svo.max_depth = depth;
    mesh.svo.debug_draw_depth = std::min(config.svo_debug_draw_depth, depth);
    mesh.svo.debug_max_boxes = std::max(1u, config.svo_debug_max_boxes);
    mesh.svo.occupied_leaf_count = static_cast<uint32_t>(keys.size());
    mesh.svo.dig_edit_count = static_cast<uint32_t>(config.voxel_edits.digs.size());
    mesh.svo.dig_removed_leaf_count = dig_removed_leaf_count;
    mesh.svo.local_edit_depth = config.voxel_edits.digs.empty() ? 0u : config.voxel_edits.local_depth;
    if (!keys.empty()) {
        mesh.svo.nodes.resize(1);
        build_svo_node_at(mesh.svo.nodes, 0, keys, 0, static_cast<uint32_t>(keys.size()), 0, 0, 0, 0, resolution, depth);
        count_svo_debug_boxes_recursive(mesh.svo, 0, mesh.svo.debug_draw_depth, mesh.svo.debug_max_boxes, mesh.svo.debug_box_count);
    }
}

} // namespace

QuantizedMesh build_quantized_marching_cubes(
    const GoldbergTopology& topology,
    const PointCloud& points,
    const MarchingCubesConfig& config
) {
    QuantizedMesh mesh;
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

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        const float surface_radius = owned_surface_radius(points, cell_id);
        const uint32_t material_id = cell.kind == GoldbergCellKind::Pentagon ? 1u : 0u;
        const uint32_t cell_subdivisions = cell_lod_subdivisions(cell, config, plane_subdivisions);
        mesh.min_cell_subdivisions = std::min(mesh.min_cell_subdivisions, cell_subdivisions);
        mesh.max_cell_subdivisions = std::max(mesh.max_cell_subdivisions, cell_subdivisions);
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
            fracture_shard_scratch
        );
    }

    if (topology.cells.empty()) {
        mesh.min_cell_subdivisions = 0;
    }
    sort_and_merge_boundary_edges(boundary_edges);
    build_transition_stitches(mesh, topology, points, boundary_edges, config, fracture_cache);
    generate_sparse_voxel_octree(mesh, config);

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
        report << '.';
    }

    return {ok, report.str()};
}

} // namespace ae
