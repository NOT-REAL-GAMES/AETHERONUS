#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/topology.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ae {

struct MarchingCubesConfig {
    uint32_t resolution_x = 8;
    uint32_t resolution_y = 8;
    uint32_t resolution_z = 4;
    float radial_half_depth = 0.08f;
    bool quantize_positions = true;
};

struct QuantizedMeshVertex {
    Vec3 position;
    Vec3 normal;
    uint32_t material_id = 0;
    uint32_t cell_id = 0;
};

struct QuantizedMesh {
    std::vector<QuantizedMeshVertex> vertices;
    std::vector<uint32_t> triangle_indices;
    std::vector<uint32_t> line_indices;
    std::vector<uint32_t> stitch_triangle_indices;
    std::vector<uint32_t> stitch_line_indices;
    uint32_t triangle_count = 0;
    uint32_t rejected_triangle_count = 0;
    uint32_t stitch_triangle_count = 0;
    uint32_t boundary_edge_count = 0;
    uint32_t chain_stitch_triangle_count = 0;
    uint32_t fallback_stitch_triangle_count = 0;
    uint32_t boundary_run_count = 0;
    uint32_t paired_boundary_run_count = 0;
    uint32_t rejected_stitch_run_count = 0;
    uint32_t unstitched_gap_count = 0;
    uint32_t clipped_triangle_count = 0;
    uint32_t discarded_clipped_triangle_count = 0;
    uint32_t shared_edge_path_count = 0;
    uint32_t greedy_path_step_count = 0;
    uint32_t rejected_greedy_jump_count = 0;
    uint32_t cell_count = 0;
};

struct QuantizedMeshValidation {
    bool ok = false;
    std::string message;
};

QuantizedMesh build_quantized_marching_cubes(
    const GoldbergTopology& topology,
    const std::vector<PointSample>& points,
    const MarchingCubesConfig& config = {}
);

QuantizedMeshValidation validate_quantized_mesh(const QuantizedMesh& mesh);

} // namespace ae
