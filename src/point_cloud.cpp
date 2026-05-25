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

void append_sample(PointCloud& points, const GoldbergTopology& topology, Vec3 position, uint32_t source_cell_id, uint32_t material_id) {
    const Vec3 normalized_position = normalize(position);
    points.positions.push_back(normalized_position);
    points.normals.push_back(normalized_position);
    points.source_cell_ids.push_back(source_cell_id);
    points.owner_cell_ids.push_back(nearest_cell_on_sphere(topology, normalized_position));
    points.material_ids.push_back(material_id);
}

void build_owned_radius_lookup(PointCloud& points, uint32_t cell_count) {
    points.owned_radius_by_cell.assign(cell_count, 1.0f);
    std::vector<float> radius_sum(cell_count, 0.0f);
    std::vector<uint32_t> radius_count(cell_count, 0);

    for (uint32_t i = 0; i < points.size(); ++i) {
        const uint32_t owner_cell_id = points.owner_cell_ids[i];
        if (owner_cell_id >= cell_count) {
            continue;
        }
        radius_sum[owner_cell_id] += length(points.positions[i]);
        ++radius_count[owner_cell_id];
    }

    for (uint32_t cell_id = 0; cell_id < cell_count; ++cell_id) {
        if (radius_count[cell_id] > 0) {
            points.owned_radius_by_cell[cell_id] = radius_sum[cell_id] / static_cast<float>(radius_count[cell_id]);
        }
    }
}

} // namespace

PointCloud build_surface_point_cloud(const GoldbergTopology& topology) {
    PointCloud points;
    const size_t sample_count = topology.cells.size() * 4;
    points.positions.reserve(sample_count);
    points.normals.reserve(sample_count);
    points.source_cell_ids.reserve(sample_count);
    points.owner_cell_ids.reserve(sample_count);
    points.material_ids.reserve(sample_count);

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        const uint32_t material = cell.kind == GoldbergCellKind::Pentagon ? 1u : 0u;

        append_sample(points, topology, cell.center, cell_id, material);

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
            append_sample(points, topology, position, cell_id, material + 2u);
        }
    }

    build_owned_radius_lookup(points, static_cast<uint32_t>(topology.cells.size()));
    return points;
}

PointCloudValidation validate_point_cloud(const GoldbergTopology& topology, const PointCloud& points) {
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
    require(points.normals.size() == points.size(), "normal count does not match point count");
    require(points.source_cell_ids.size() == points.size(), "source cell id count does not match point count");
    require(points.owner_cell_ids.size() == points.size(), "owner cell id count does not match point count");
    require(points.material_ids.size() == points.size(), "material id count does not match point count");
    require(points.owned_radius_by_cell.size() == topology.cells.size(), "owned radius lookup count does not match cell count");

    for (uint32_t i = 0; i < points.size(); ++i) {
        require(points.source_cell_ids[i] < topology.cells.size(), "source cell id is out of range");
        require(points.owner_cell_ids[i] < topology.cells.size(), "owner cell id is out of range");
        require(std::fabs(length(points.positions[i]) - 1.0f) < 0.0005f, "point position is not on the unit sphere");

        if (points.source_cell_ids[i] != points.owner_cell_ids[i]) {
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
