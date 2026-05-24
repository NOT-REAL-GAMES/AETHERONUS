#include "aetheronus/meshing.hpp"
#include "aetheronus/marching_cubes_tables.hpp"

#include <algorithm>
#include <array>
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

struct PositionKey {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
};

struct EdgeKey {
    PositionKey a;
    PositionKey b;
};

struct CellEdgeKey {
    uint32_t cell_id = 0;
    EdgeKey edge;
};

struct BoundaryEdgeRecord {
    uint32_t cell_id = 0;
    Vec3 a;
    Vec3 b;
    uint32_t count = 0;
};

struct BoundarySegment {
    Vec3 a;
    Vec3 b;
    Vec3 midpoint;
    float sort_value = 0.0f;
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
};

using BoundaryEdgeMap = std::map<CellEdgeKey, BoundaryEdgeRecord>;
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

Frame build_frame(Vec3 normal) {
    const Vec3 reference_axis = std::fabs(normal.y) < 0.92f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    const Vec3 tangent = normalize(cross(reference_axis, normal));
    const Vec3 bitangent = cross(normal, tangent);
    return {tangent, bitangent, normal};
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

float owned_surface_radius(const std::vector<PointSample>& points, uint32_t cell_id) {
    float radius_sum = 0.0f;
    uint32_t count = 0;
    for (const PointSample& point : points) {
        if (point.owner_cell_id == cell_id) {
            radius_sum += length(point.position);
            ++count;
        }
    }
    return count > 0 ? radius_sum / static_cast<float>(count) : 1.0f;
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

void record_triangle_edge(BoundaryEdgeMap& edges, uint32_t cell_id, Vec3 a, Vec3 b) {
    const CellEdgeKey key = {cell_id, edge_key(a, b)};
    BoundaryEdgeRecord& record = edges[key];
    if (record.count == 0) {
        record.cell_id = cell_id;
        record.a = a;
        record.b = b;
    }
    ++record.count;
}

void record_triangle_edges(BoundaryEdgeMap& edges, uint32_t cell_id, Vec3 a, Vec3 b, Vec3 c) {
    record_triangle_edge(edges, cell_id, a, b);
    record_triangle_edge(edges, cell_id, b, c);
    record_triangle_edge(edges, cell_id, c, a);
}

void emit_mesh_triangle(
    QuantizedMesh& mesh,
    BoundaryEdgeMap& boundary_edges,
    uint32_t emitting_cell_id,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    uint32_t material_id
) {
    const Vec3 normal = normalize(cross(b - a, c - a));
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back({a, normal, material_id, emitting_cell_id});
    mesh.vertices.push_back({b, normal, material_id, emitting_cell_id});
    mesh.vertices.push_back({c, normal, material_id, emitting_cell_id});
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
    record_triangle_edges(boundary_edges, emitting_cell_id, a, b, c);
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
    uint32_t material_id
) {
    if (dot(cross(b - a, c - a), outward) < 0.0f) {
        std::swap(b, c);
    }
    emit_mesh_triangle(mesh, boundary_edges, emitting_cell_id, a, b, c, material_id);
}

Vec3 subdivided_cell_vertex(Vec3 center, Vec3 edge_a, Vec3 edge_b, uint32_t a_step, uint32_t b_step, uint32_t subdivisions, float radius) {
    const float inv_subdivisions = 1.0f / static_cast<float>(subdivisions);
    const float a_weight = static_cast<float>(a_step) * inv_subdivisions;
    const float b_weight = static_cast<float>(b_step) * inv_subdivisions;
    const float center_weight = 1.0f - a_weight - b_weight;
    return normalize(center * center_weight + edge_a * a_weight + edge_b * b_weight) * radius;
}

void emit_subdivided_goldberg_cell_plane(
    QuantizedMesh& mesh,
    BoundaryEdgeMap& boundary_edges,
    const GoldbergTopology& topology,
    uint32_t cell_id,
    uint32_t subdivisions,
    float surface_radius,
    uint32_t material_id
) {
    const GoldbergCell& cell = topology.cells[cell_id];
    if (cell.corner_indices.size() < 3 || subdivisions == 0) {
        return;
    }

    for (uint32_t corner_slot = 0; corner_slot < cell.corner_indices.size(); ++corner_slot) {
        const uint32_t next_slot = (corner_slot + 1) % static_cast<uint32_t>(cell.corner_indices.size());
        const Vec3 center = cell.center;
        const Vec3 edge_a = topology.vertices[cell.corner_indices[corner_slot]].position;
        const Vec3 edge_b = topology.vertices[cell.corner_indices[next_slot]].position;

        for (uint32_t a_step = 0; a_step < subdivisions; ++a_step) {
            for (uint32_t b_step = 0; b_step + a_step < subdivisions; ++b_step) {
                const Vec3 p0 = subdivided_cell_vertex(center, edge_a, edge_b, a_step, b_step, subdivisions, surface_radius);
                const Vec3 p1 = subdivided_cell_vertex(center, edge_a, edge_b, a_step + 1, b_step, subdivisions, surface_radius);
                const Vec3 p2 = subdivided_cell_vertex(center, edge_a, edge_b, a_step, b_step + 1, subdivisions, surface_radius);
                emit_oriented_mesh_triangle(mesh, boundary_edges, cell_id, p0, p1, p2, cell.normal, material_id);

                if (a_step + b_step + 1 < subdivisions) {
                    const Vec3 p3 = subdivided_cell_vertex(center, edge_a, edge_b, a_step + 1, b_step + 1, subdivisions, surface_radius);
                    emit_oriented_mesh_triangle(mesh, boundary_edges, cell_id, p1, p3, p2, cell.normal, material_id);
                }
            }
        }
    }
}

void append_stitch_triangle(QuantizedMesh& mesh, Vec3 a, Vec3 b, Vec3 c, uint32_t cell_id) {
    const Vec3 normal = normalize(cross(b - a, c - a));
    const uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    constexpr uint32_t TransitionMaterial = 2u;

    mesh.vertices.push_back({a, normal, TransitionMaterial, cell_id});
    mesh.vertices.push_back({b, normal, TransitionMaterial, cell_id});
    mesh.vertices.push_back({c, normal, TransitionMaterial, cell_id});

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

void append_chain_stitch_triangle(QuantizedMesh& mesh, Vec3 a, Vec3 b, Vec3 c, uint32_t cell_id) {
    append_stitch_triangle(mesh, a, b, c, cell_id);
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

Vec3 stitch_side_sample(Vec3 edge_point, Vec3 cell_center, float radius) {
    constexpr float StripHalfWidth = 0.035f;
    return normalize(lerp(edge_point, cell_center, StripHalfWidth)) * radius;
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

bool append_ideal_edge_strip(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const std::vector<PointSample>& points,
    uint32_t cell_id,
    uint32_t neighbor_id
) {
    constexpr uint32_t SamplesPerEdge = 6;

    const GoldbergCell& cell = topology.cells[cell_id];
    const GoldbergCell& neighbor = topology.cells[neighbor_id];
    const std::array<uint32_t, 2> shared_edge = shared_goldberg_edge(cell, neighbor);
    if (shared_edge[0] == UINT32_MAX || shared_edge[1] == UINT32_MAX) {
        return false;
    }

    const Vec3 center_a = topology.cells[cell_id].center;
    const Vec3 center_b = topology.cells[neighbor_id].center;
    Vec3 edge_a = topology.vertices[shared_edge[0]].position;
    Vec3 edge_b = topology.vertices[shared_edge[1]].position;
    const Vec3 edge_mid = normalize(edge_a + edge_b);
    const Vec3 center_mid = normalize(center_a + center_b);
    if (dot(edge_mid, center_mid) < 0.0f) {
        std::swap(edge_a, edge_b);
    }

    const float radius = (owned_surface_radius(points, cell_id) + owned_surface_radius(points, neighbor_id)) * 0.5f;

    for (uint32_t i = 0; i + 1 < SamplesPerEdge; ++i) {
        const float t0 = static_cast<float>(i) / static_cast<float>(SamplesPerEdge - 1);
        const float t1 = static_cast<float>(i + 1) / static_cast<float>(SamplesPerEdge - 1);
        const Vec3 edge0 = normalize(lerp(edge_a, edge_b, t0));
        const Vec3 edge1 = normalize(lerp(edge_a, edge_b, t1));
        const Vec3 a0 = stitch_side_sample(edge0, center_a, radius);
        const Vec3 a1 = stitch_side_sample(edge1, center_a, radius);
        const Vec3 b0 = stitch_side_sample(edge0, center_b, radius);
        const Vec3 b1 = stitch_side_sample(edge1, center_b, radius);

        append_fallback_stitch_triangle(mesh, a0, b0, a1, cell_id);
        append_fallback_stitch_triangle(mesh, a1, b0, b1, cell_id);
    }

    return true;
}

BoundaryPairMap build_boundary_pair_chains(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const BoundaryEdgeMap& boundary_edges
) {
    BoundaryPairMap pairs;

    for (const auto& entry : boundary_edges) {
        const BoundaryEdgeRecord& record = entry.second;
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
        BoundarySegment segment = {record.a, record.b, midpoint, 0.0f};
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
    std::map<PositionKey, CorridorPoint> unique_points;
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

            unique_points[position_key(endpoint)] = {endpoint, sort_value};
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
        });
        b_start = best_b + 1;
    }

    if (pairs.size() < 2) {
        return false;
    }

    const float max_path_step = std::max(geometry.edge_length * 0.30f, 0.040f);
    const float max_triangle_edge = std::max(geometry.edge_length * 0.46f, 0.060f);
    bool emitted = false;

    for (uint32_t i = 0; i + 1 < pairs.size(); ++i) {
        const Vec3 a0 = pairs[i].a;
        const Vec3 a1 = pairs[i + 1].a;
        const Vec3 b0 = pairs[i].b;
        const Vec3 b1 = pairs[i + 1].b;

        const bool path_ok =
            length(a0 - a1) <= max_path_step &&
            length(b0 - b1) <= max_path_step &&
            pairs[i + 1].sort_value >= pairs[i].sort_value;
        const bool edges_ok =
            length(a1 - b0) <= max_triangle_edge &&
            length(a0 - b1) <= max_triangle_edge;

        if (!path_ok || !edges_ok) {
            ++mesh.rejected_greedy_jump_count;
            continue;
        }

        append_chain_stitch_triangle(mesh, normalize(a0) * radius, normalize(b0) * radius, normalize(a1) * radius, cell_id);
        append_chain_stitch_triangle(mesh, normalize(a1) * radius, normalize(b0) * radius, normalize(b1) * radius, cell_id);
        ++mesh.greedy_path_step_count;
        emitted = true;
    }

    return emitted;
}

void append_chain_stitches_for_pair(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const std::vector<PointSample>& points,
    const BoundaryPairKey& pair_key,
    const BoundaryPairChains& chains
) {
    BoundaryPairGeometry geometry;
    if (!shared_boundary_geometry(topology, pair_key.a, pair_key.b, geometry)) {
        return;
    }

    const float radius = (owned_surface_radius(points, pair_key.a) + owned_surface_radius(points, pair_key.b)) * 0.5f;
    std::vector<CorridorPoint> side_a = collect_corridor_points(chains.a_segments, geometry, radius);
    std::vector<CorridorPoint> side_b = collect_corridor_points(chains.b_segments, geometry, radius);

    if (side_a.size() < 2 || side_b.size() < 2) {
        ++mesh.unstitched_gap_count;
        return;
    }

    ++mesh.shared_edge_path_count;
    ++mesh.boundary_run_count;
    ++mesh.paired_boundary_run_count;
    if (!append_greedy_corridor_stitches(mesh, side_a, side_b, geometry, pair_key.a, radius)) {
        ++mesh.rejected_stitch_run_count;
        ++mesh.unstitched_gap_count;
    }
}

void build_transition_stitches(
    QuantizedMesh& mesh,
    const GoldbergTopology& topology,
    const std::vector<PointSample>& points,
    const BoundaryEdgeMap& boundary_edges
) {
    BoundaryPairMap boundary_pairs = build_boundary_pair_chains(mesh, topology, boundary_edges);

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

            append_chain_stitches_for_pair(mesh, topology, points, pair_key, found->second);
        }
    }
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

} // namespace

QuantizedMesh build_quantized_marching_cubes(
    const GoldbergTopology& topology,
    const std::vector<PointSample>& points,
    const MarchingCubesConfig& config
) {
    QuantizedMesh mesh;
    mesh.cell_count = static_cast<uint32_t>(topology.cells.size());
    BoundaryEdgeMap boundary_edges;
    const uint32_t plane_subdivisions = std::max(1u, std::max(config.resolution_x, config.resolution_y));

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        const float surface_radius = owned_surface_radius(points, cell_id);
        const uint32_t material_id = cell.kind == GoldbergCellKind::Pentagon ? 1u : 0u;
        emit_subdivided_goldberg_cell_plane(mesh, boundary_edges, topology, cell_id, plane_subdivisions, surface_radius, material_id);
    }

    build_transition_stitches(mesh, topology, points, boundary_edges);

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

    if (ok) {
        report << "Goldberg plane mesh OK: " << mesh.cell_count << " cells, "
               << mesh.vertices.size() << " vertices, "
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
               << mesh.unstitched_gap_count << " unstitched gaps.";
    }

    return {ok, report.str()};
}

} // namespace ae
