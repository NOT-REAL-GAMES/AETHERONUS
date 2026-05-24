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

struct PointCloudValidation {
    bool ok = false;
    uint32_t mismatch_count = 0;
    std::string message;
};

std::vector<PointSample> build_surface_point_cloud(const GoldbergTopology& topology);
PointCloudValidation validate_point_cloud(const GoldbergTopology& topology, const std::vector<PointSample>& points);

} // namespace ae
