#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/topology.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ae {

struct PointSample {
    Vec3 position;
    Vec3 normal;
    uint32_t source_cell_id = 0;
    uint32_t owner_cell_id = 0;
    uint32_t material_id = 0;
};

struct PointCloud {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<uint32_t> source_cell_ids;
    std::vector<uint32_t> owner_cell_ids;
    std::vector<uint32_t> material_ids;
    std::vector<float> owned_radius_by_cell;

    size_t size() const {
        return positions.size();
    }

    bool empty() const {
        return positions.empty();
    }
};

struct PointCloudValidation {
    bool ok = false;
    uint32_t mismatch_count = 0;
    std::string message;
};

PointCloud build_surface_point_cloud(const GoldbergTopology& topology);
PointCloudValidation validate_point_cloud(const GoldbergTopology& topology, const PointCloud& points);

} // namespace ae
