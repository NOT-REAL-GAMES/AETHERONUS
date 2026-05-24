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
    bool enable_lod_subdivision_test = false;
    bool enable_camera_proximity_lod = false;
    uint32_t lod_min_subdivisions = 4;
    uint32_t lod_max_subdivisions = 16;
    uint32_t lod_levels = 4;
    Vec3 lod_camera_position = {0.0f, 0.0f, 3.4f};
    float lod_inner_patch_radius = 0.18f;
    float lod_outer_patch_radius = 0.95f;
    bool enable_fractures = false;
    bool connect_fractures_across_cells = true;
    uint32_t fracture_seed = 1;
    uint32_t global_fracture_seed_copies = 1;
    uint32_t shards_per_hex = 7;
    uint32_t shards_per_pent = 6;
    float fracture_gap = 0.024f;
    float fracture_depth = 0.010f;
    float fracture_edge_guard = 0.0030f;
    bool enable_fracture_walls = true;
    float fracture_wall_depth = 0.30f;
    uint32_t fracture_wall_material_id = 4;
    float fracture_chunk_outward_min = 0.030f;
    float fracture_chunk_outward_max = 0.120f;
};

struct QuantizedMeshVertex {
    Vec3 position;
    Vec3 normal;
    uint32_t material_id = 0;
    uint32_t cell_id = 0;
    uint32_t fracture_chunk_id = 0;
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
    uint32_t min_cell_subdivisions = 0;
    uint32_t max_cell_subdivisions = 0;
    uint32_t lod_level_count = 0;
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
