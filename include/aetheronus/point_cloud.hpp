#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/topology.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ae {

enum class PointSampleKind : uint8_t {
    Center,
    Corner,
    Edge,
    Spoke,
    Interior,
};

struct PointSample {
    Vec3 position;
    Vec3 normal;
    uint32_t source_cell_id = 0;
    uint32_t owner_cell_id = 0;
    uint32_t material_id = 0;
    PointSampleKind kind = PointSampleKind::Interior;
    uint32_t deterministic_id = 0;
    std::vector<uint32_t> participating_cell_ids;
};

struct PointTriangle {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t cell_id = 0;
    uint32_t material_id = 0;
};

struct PointCloudConfig {
    uint32_t radial_subdivisions = 16;
    uint32_t edge_subdivisions = 16;
};

struct PointCloud {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<uint32_t> source_cell_ids;
    std::vector<uint32_t> owner_cell_ids;
    std::vector<uint32_t> material_ids;
    std::vector<float> owned_radius_by_cell;
    std::vector<PointSampleKind> sample_kinds;
    std::vector<uint32_t> deterministic_ids;
    std::vector<std::vector<uint32_t>> participating_cell_ids;
    std::vector<PointTriangle> triangles;
    PointCloudConfig config;

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

PointCloud build_surface_point_cloud(const GoldbergTopology& topology, PointCloudConfig config = {});
PointCloudValidation validate_point_cloud(const GoldbergTopology& topology, const PointCloud& points);

} // namespace ae
