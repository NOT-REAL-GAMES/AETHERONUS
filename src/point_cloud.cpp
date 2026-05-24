#include "aetheronus/point_cloud.hpp"

#include <array>
#include <cmath>
#include <sstream>

namespace ae {
namespace {

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

PointSample make_sample(const GoldbergTopology& topology, Vec3 position, uint32_t source_cell_id, uint32_t material_id) {
    const Vec3 normalized_position = normalize(position);
    return {
        normalized_position,
        normalized_position,
        source_cell_id,
        nearest_cell_on_sphere(topology, normalized_position),
        material_id,
    };
}

} // namespace

std::vector<PointSample> build_surface_point_cloud(const GoldbergTopology& topology) {
    std::vector<PointSample> points;
    points.reserve(topology.cells.size() * 4);

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        const uint32_t material = cell.kind == GoldbergCellKind::Pentagon ? 1u : 0u;

        points.push_back(make_sample(topology, cell.center, cell_id, material));

        const uint32_t corner_count = static_cast<uint32_t>(cell.corner_indices.size());
        const std::array<uint32_t, 3> sample_corners = {
            0u,
            corner_count / 3u,
            (corner_count * 2u) / 3u,
        };

        for (uint32_t corner_slot : sample_corners) {
            const uint32_t corner_index = cell.corner_indices[corner_slot % corner_count];
            const Vec3 corner = topology.vertices[corner_index].position;
            const Vec3 position = normalize(lerp(cell.center, corner, 0.38f));
            points.push_back(make_sample(topology, position, cell_id, material + 2u));
        }
    }

    return points;
}

PointCloudValidation validate_point_cloud(const GoldbergTopology& topology, const std::vector<PointSample>& points) {
    std::ostringstream report;
    bool ok = true;
    uint32_t mismatch_count = 0;

    auto require = [&](bool condition, const char* message) {
        if (!condition) {
            ok = false;
            report << "Point assignment failed: " << message << '\n';
        }
    };

    require(points.size() == topology.cells.size() * 4, "point count is not cell_count * 4");

    for (const PointSample& point : points) {
        require(point.source_cell_id < topology.cells.size(), "source cell id is out of range");
        require(point.owner_cell_id < topology.cells.size(), "owner cell id is out of range");
        require(std::fabs(length(point.position) - 1.0f) < 0.0005f, "point position is not on the unit sphere");

        if (point.source_cell_id != point.owner_cell_id) {
            ++mismatch_count;
        }
    }

    if (ok) {
        report << "Point assignment OK: " << points.size() << " samples, "
               << mismatch_count << " ownership mismatches.";
    }

    return {ok, mismatch_count, report.str()};
}

} // namespace ae
