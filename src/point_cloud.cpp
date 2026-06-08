#include "aetheronus/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <unordered_map>

namespace ae {
namespace {

struct BoundarySampleKey {
    uint32_t kind = 0;
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t step = 0;
    uint32_t denominator = 1;

    bool operator==(const BoundarySampleKey& rhs) const {
        return kind == rhs.kind &&
               a == rhs.a &&
               b == rhs.b &&
               step == rhs.step &&
               denominator == rhs.denominator;
    }
};

struct BoundarySampleKeyHash {
    size_t operator()(BoundarySampleKey key) const {
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&](uint32_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(key.kind);
        mix(key.a);
        mix(key.b);
        mix(key.step);
        mix(key.denominator);
        return static_cast<size_t>(hash);
    }
};

void add_unique_participant(std::vector<uint32_t>& participants, uint32_t cell_id) {
    if (std::find(participants.begin(), participants.end(), cell_id) == participants.end()) {
        participants.push_back(cell_id);
    }
}

uint32_t sample_material_id(const GoldbergTopology& topology, uint32_t cell_id, PointSampleKind kind) {
    if (kind == PointSampleKind::Center) {
        return topology.cells[cell_id].kind == GoldbergCellKind::Pentagon ? 1u : 0u;
    }
    switch (kind) {
        case PointSampleKind::Corner:
            return 2u;
        case PointSampleKind::Edge:
            return 3u;
        case PointSampleKind::Spoke:
            return 4u;
        case PointSampleKind::Interior:
            return 5u;
        default:
            return 0u;
    }
}

uint32_t nearest_cell_on_sphere(const GoldbergTopology& topology, Vec3 position) {
    uint32_t nearest = 0;
    float best_dot = -2.0f;
    const Vec3 direction = normalize(position);

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const float score = dot(direction, topology.cells[cell_id].center);
        if (score > best_dot) {
            best_dot = score;
            nearest = cell_id;
        }
    }

    return nearest;
}

uint32_t append_sample(
    PointCloud& points,
    const GoldbergTopology& topology,
    Vec3 position,
    uint32_t source_cell_id,
    PointSampleKind kind,
    std::vector<uint32_t> participants
) {
    const Vec3 normalized = normalize(position);
    const uint32_t owner_cell_id = nearest_cell_on_sphere(topology, normalized);
    const uint32_t index = static_cast<uint32_t>(points.positions.size());
    if (participants.empty()) {
        participants.push_back(source_cell_id);
    }
    add_unique_participant(participants, source_cell_id);
    std::sort(participants.begin(), participants.end());

    points.positions.push_back(normalized);
    points.normals.push_back(normalized);
    points.source_cell_ids.push_back(source_cell_id);
    points.owner_cell_ids.push_back(owner_cell_id);
    points.material_ids.push_back(sample_material_id(topology, source_cell_id, kind));
    points.sample_kinds.push_back(kind);
    points.deterministic_ids.push_back(index);
    points.participating_cell_ids.push_back(std::move(participants));
    return index;
}

BoundarySampleKey corner_key(uint32_t corner_index) {
    return {0u, corner_index, 0u, 0u, 1u};
}

BoundarySampleKey edge_key(uint32_t corner_a, uint32_t corner_b, uint32_t step, uint32_t denominator) {
    if (corner_b < corner_a) {
        std::swap(corner_a, corner_b);
        step = denominator - step;
    }
    return {1u, corner_a, corner_b, step, denominator};
}

std::vector<uint32_t> corner_participants(const GoldbergTopology& topology, uint32_t corner_index) {
    std::vector<uint32_t> participants;
    if (corner_index >= topology.geodesic.triangles.size()) {
        return participants;
    }

    const Triangle& triangle = topology.geodesic.triangles[corner_index];
    participants = {triangle.a, triangle.b, triangle.c};
    std::sort(participants.begin(), participants.end());
    participants.erase(std::unique(participants.begin(), participants.end()), participants.end());
    return participants;
}

uint32_t shared_boundary_sample(
    PointCloud& points,
    const GoldbergTopology& topology,
    std::unordered_map<BoundarySampleKey, uint32_t, BoundarySampleKeyHash>& lookup,
    BoundarySampleKey key,
    Vec3 position,
    uint32_t cell_id,
    PointSampleKind kind,
    std::vector<uint32_t> participants
) {
    const auto found = lookup.find(key);
    if (found != lookup.end()) {
        add_unique_participant(points.participating_cell_ids[found->second], cell_id);
        return found->second;
    }

    const uint32_t index = append_sample(points, topology, position, cell_id, kind, std::move(participants));
    lookup.emplace(key, index);
    return index;
}

std::vector<uint32_t> build_boundary_ring(
    PointCloud& points,
    const GoldbergTopology& topology,
    std::unordered_map<BoundarySampleKey, uint32_t, BoundarySampleKeyHash>& shared_lookup,
    uint32_t cell_id,
    uint32_t edge_subdivisions
) {
    const GoldbergCell& cell = topology.cells[cell_id];
    std::vector<uint32_t> ring;
    const uint32_t corner_count = static_cast<uint32_t>(cell.corner_indices.size());
    ring.reserve(static_cast<size_t>(corner_count) * edge_subdivisions);

    for (uint32_t corner_slot = 0; corner_slot < corner_count; ++corner_slot) {
        const uint32_t next_slot = (corner_slot + 1u) % corner_count;
        const uint32_t corner_a = cell.corner_indices[corner_slot];
        const uint32_t corner_b = cell.corner_indices[next_slot];
        const Vec3 a = topology.vertices[corner_a].position;
        const Vec3 b = topology.vertices[corner_b].position;

        for (uint32_t step = 0; step < edge_subdivisions; ++step) {
            if (step == 0u) {
                ring.push_back(shared_boundary_sample(
                    points,
                    topology,
                    shared_lookup,
                    corner_key(corner_a),
                    a,
                    cell_id,
                    PointSampleKind::Corner,
                    corner_participants(topology, corner_a)
                ));
                continue;
            }

            const float t = static_cast<float>(step) / static_cast<float>(edge_subdivisions);
            ring.push_back(shared_boundary_sample(
                points,
                topology,
                shared_lookup,
                edge_key(corner_a, corner_b, step, edge_subdivisions),
                normalize(lerp(a, b, t)),
                cell_id,
                PointSampleKind::Edge,
                {cell_id}
            ));
        }
    }

    return ring;
}

uint32_t append_cell_local_sample(
    PointCloud& points,
    const GoldbergTopology& topology,
    uint32_t cell_id,
    Vec3 position,
    PointSampleKind kind
) {
    return append_sample(points, topology, position, cell_id, kind, {cell_id});
}

void append_triangle(PointCloud& points, uint32_t a, uint32_t b, uint32_t c, uint32_t cell_id, uint32_t material_id) {
    if (a == b || b == c || c == a) {
        return;
    }
    points.triangles.push_back({a, b, c, cell_id, material_id});
}

void build_cell_lattice(
    PointCloud& points,
    const GoldbergTopology& topology,
    std::unordered_map<BoundarySampleKey, uint32_t, BoundarySampleKeyHash>& shared_lookup,
    uint32_t cell_id
) {
    const GoldbergCell& cell = topology.cells[cell_id];
    if (cell.corner_indices.size() < 3) {
        return;
    }

    const uint32_t radial_subdivisions = std::max(1u, points.config.radial_subdivisions);
    const uint32_t edge_subdivisions = std::max(1u, points.config.edge_subdivisions);
    const uint32_t material_id = cell.kind == GoldbergCellKind::Pentagon ? 1u : 0u;
    const uint32_t center_index = append_cell_local_sample(points, topology, cell_id, cell.center, PointSampleKind::Center);
    const std::vector<uint32_t> boundary_ring = build_boundary_ring(points, topology, shared_lookup, cell_id, edge_subdivisions);
    const uint32_t ring_count = static_cast<uint32_t>(boundary_ring.size());
    if (ring_count < 3u) {
        return;
    }

    std::vector<std::vector<uint32_t>> rings;
    rings.reserve(radial_subdivisions + 1u);
    rings.push_back({center_index});

    for (uint32_t radial = 1u; radial < radial_subdivisions; ++radial) {
        const float t = static_cast<float>(radial) / static_cast<float>(radial_subdivisions);
        std::vector<uint32_t> ring;
        ring.reserve(ring_count);
        for (uint32_t slot = 0; slot < ring_count; ++slot) {
            const Vec3 boundary = points.positions[boundary_ring[slot]];
            const bool spoke = (slot % edge_subdivisions) == 0u;
            ring.push_back(append_cell_local_sample(
                points,
                topology,
                cell_id,
                normalize(lerp(cell.center, boundary, t)),
                spoke ? PointSampleKind::Spoke : PointSampleKind::Interior
            ));
        }
        rings.push_back(std::move(ring));
    }
    rings.push_back(boundary_ring);

    const std::vector<uint32_t>& first_ring = rings[1u];
    for (uint32_t slot = 0; slot < ring_count; ++slot) {
        append_triangle(
            points,
            center_index,
            first_ring[slot],
            first_ring[(slot + 1u) % ring_count],
            cell_id,
            material_id
        );
    }

    for (uint32_t radial = 1u; radial < radial_subdivisions; ++radial) {
        const std::vector<uint32_t>& inner = rings[radial];
        const std::vector<uint32_t>& outer = rings[radial + 1u];
        for (uint32_t slot = 0; slot < ring_count; ++slot) {
            const uint32_t next = (slot + 1u) % ring_count;
            append_triangle(points, inner[slot], outer[slot], outer[next], cell_id, material_id);
            append_triangle(points, inner[slot], outer[next], inner[next], cell_id, material_id);
        }
    }
}

void build_owned_radius_lookup(PointCloud& points, uint32_t cell_count) {
    points.owned_radius_by_cell.assign(cell_count, 1.0f);
}

} // namespace

PointCloud build_surface_point_cloud(const GoldbergTopology& topology, PointCloudConfig config) {
    PointCloud points;
    config.radial_subdivisions = std::max(1u, config.radial_subdivisions);
    config.edge_subdivisions = std::max(1u, config.edge_subdivisions);
    points.config = config;

    const size_t approximate_per_hex =
        1u + static_cast<size_t>(6u * config.edge_subdivisions) * static_cast<size_t>(config.radial_subdivisions);
    points.positions.reserve(topology.cells.size() * approximate_per_hex);
    points.normals.reserve(topology.cells.size() * approximate_per_hex);
    points.source_cell_ids.reserve(topology.cells.size() * approximate_per_hex);
    points.owner_cell_ids.reserve(topology.cells.size() * approximate_per_hex);
    points.material_ids.reserve(topology.cells.size() * approximate_per_hex);
    points.sample_kinds.reserve(topology.cells.size() * approximate_per_hex);
    points.deterministic_ids.reserve(topology.cells.size() * approximate_per_hex);
    points.participating_cell_ids.reserve(topology.cells.size() * approximate_per_hex);
    points.triangles.reserve(topology.cells.size() * approximate_per_hex * 2u);

    std::unordered_map<BoundarySampleKey, uint32_t, BoundarySampleKeyHash> shared_lookup;
    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        build_cell_lattice(points, topology, shared_lookup, cell_id);
    }

    build_owned_radius_lookup(points, static_cast<uint32_t>(topology.cells.size()));
    return points;
}

PointCloudValidation validate_point_cloud(const GoldbergTopology& topology, const PointCloud& points) {
    std::ostringstream report;
    bool ok = true;
    uint32_t mismatch_count = 0;
    uint32_t center_count = 0;
    uint32_t corner_count = 0;
    uint32_t edge_count = 0;
    uint32_t spoke_count = 0;
    uint32_t interior_count = 0;

    auto require = [&](bool condition, const char* message) {
        if (!condition) {
            ok = false;
            report << "Point lattice failed: " << message << '\n';
        }
    };

    require(points.size() >= topology.cells.size(), "point count is less than cell count");
    require(!points.triangles.empty(), "point triangle buffer is empty");
    require(points.normals.size() == points.size(), "normal count does not match point count");
    require(points.source_cell_ids.size() == points.size(), "source cell id count does not match point count");
    require(points.owner_cell_ids.size() == points.size(), "owner cell id count does not match point count");
    require(points.material_ids.size() == points.size(), "material id count does not match point count");
    require(points.sample_kinds.size() == points.size(), "sample kind count does not match point count");
    require(points.deterministic_ids.size() == points.size(), "deterministic id count does not match point count");
    require(points.participating_cell_ids.size() == points.size(), "participating cell metadata count does not match point count");
    require(points.owned_radius_by_cell.size() == topology.cells.size(), "owned radius lookup count does not match cell count");

    std::vector<uint32_t> centers_by_cell(topology.cells.size(), 0u);
    std::vector<uint32_t> triangles_by_cell(topology.cells.size(), 0u);

    for (uint32_t i = 0; i < points.size(); ++i) {
        require(points.source_cell_ids[i] < topology.cells.size(), "source cell id is out of range");
        require(points.owner_cell_ids[i] < topology.cells.size(), "owner cell id is out of range");
        require(std::fabs(length(points.positions[i]) - 1.0f) < 0.0005f, "point position is not on the unit sphere");
        require(std::fabs(length(points.normals[i]) - 1.0f) < 0.0005f, "point normal is not unit length");

        if (points.source_cell_ids[i] != points.owner_cell_ids[i]) {
            ++mismatch_count;
        }
        if (points.source_cell_ids[i] < centers_by_cell.size() && points.sample_kinds[i] == PointSampleKind::Center) {
            ++centers_by_cell[points.source_cell_ids[i]];
        }
        for (uint32_t participant : points.participating_cell_ids[i]) {
            require(participant < topology.cells.size(), "participating cell id is out of range");
        }

        switch (points.sample_kinds[i]) {
            case PointSampleKind::Center:
                ++center_count;
                break;
            case PointSampleKind::Corner:
                ++corner_count;
                break;
            case PointSampleKind::Edge:
                ++edge_count;
                break;
            case PointSampleKind::Spoke:
                ++spoke_count;
                break;
            case PointSampleKind::Interior:
                ++interior_count;
                break;
        }
    }

    for (const PointTriangle& triangle : points.triangles) {
        require(triangle.a < points.size(), "point triangle index A is out of range");
        require(triangle.b < points.size(), "point triangle index B is out of range");
        require(triangle.c < points.size(), "point triangle index C is out of range");
        require(triangle.cell_id < topology.cells.size(), "point triangle cell id is out of range");
        if (triangle.cell_id < triangles_by_cell.size()) {
            ++triangles_by_cell[triangle.cell_id];
        }
    }

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        require(centers_by_cell[cell_id] == 1u, "cell does not have exactly one center sample");
        require(triangles_by_cell[cell_id] > 0u, "cell does not emit point lattice triangles");
    }

    if (ok) {
        report << "Point lattice OK: " << points.size() << " samples, "
               << points.triangles.size() << " lattice triangles, "
               << center_count << " centers, "
               << corner_count << " shared corners, "
               << edge_count << " shared edge samples, "
               << spoke_count << " spokes, "
               << interior_count << " interior samples, "
               << mismatch_count << " nearest-owner mismatches.";
    }

    return {ok, mismatch_count, report.str()};
}

} // namespace ae
