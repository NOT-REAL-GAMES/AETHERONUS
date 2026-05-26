#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/morton.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/topology.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ae {

struct VoxelDigEdit {
    Vec3 center_mesh;
    float radius_km = 8.0f;
    uint32_t depth = 16;
};

struct VoxelEditSet {
    std::vector<VoxelDigEdit> digs;
    uint32_t local_depth = 16;
};

struct VoxelKey {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;

    bool operator<(const VoxelKey& rhs) const {
        const uint64_t lhs_code = morton_encode(x, y, z);
        const uint64_t rhs_code = morton_encode(rhs.x, rhs.y, rhs.z);
        if (lhs_code != rhs_code) {
            return lhs_code < rhs_code;
        }
        if (x != rhs.x) return x < rhs.x;
        if (y != rhs.y) return y < rhs.y;
        return z < rhs.z;
    }

    bool operator==(const VoxelKey& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

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
    uint32_t global_fracture_seed_count = 48;
    uint32_t global_fracture_seed_copies = 1;
    uint32_t shards_per_hex = 7;
    uint32_t shards_per_pent = 6;
    float fracture_gap = 0.014f;
    float fracture_depth = 0.006f;
    float fracture_edge_guard = 0.0f;
    float fracture_boundary_falloff = 0.0f;
    bool fracture_seams_use_max_lift = false;
    bool enable_fracture_walls = true;
    float fracture_wall_depth = 0.30f;
    uint32_t fracture_wall_material_id = 4;
    float fracture_chunk_outward_min = 0.004f;
    float fracture_chunk_outward_max = 0.024f;
    bool enable_svo_generation = true;
    uint32_t svo_depth = 8;
    uint32_t svo_debug_draw_depth = 8;
    uint32_t svo_debug_max_boxes = 2000000;
    bool enable_surface_net_generation = true;
    bool enable_surface_net_dual_contouring = true;
    uint32_t surface_net_depth = 8;
    uint32_t surface_net_max_vertices = 2000000;
    uint32_t surface_net_material_id = 5;
    bool enable_local_surface_net_detail = true;
    uint32_t local_surface_net_depth = 16;
    float local_surface_net_patch_radius_km = 160.0f;
    float local_surface_net_patch_overlap_km = 24.0f;
    uint32_t local_surface_net_max_patches = 8;
    VoxelEditSet voxel_edits;
    std::function<void(double, const char*)> progress_callback;
};

struct QuantizedMeshVertex {
    Vec3 position;
    Vec3 normal;
    uint32_t material_id = 0;
    uint32_t cell_id = 0;
    uint32_t fracture_chunk_id = 0;
};

struct SparseVoxelOctreeNode {
    uint32_t child_mask = 0;
    uint32_t first_child = 0;
    uint32_t occupied_leaf_count = 0;
    uint32_t depth = 0;
    uint32_t origin_x = 0;
    uint32_t origin_y = 0;
    uint32_t origin_z = 0;
    uint32_t size = 0;
};

struct SparseVoxelOctree {
    std::vector<SparseVoxelOctreeNode> nodes;
    float bounds_radius = 0.0f;
    uint32_t depth = 0;
    uint32_t occupied_leaf_count = 0;
    uint32_t max_depth = 0;
    uint32_t debug_draw_depth = 0;
    uint32_t debug_max_boxes = 0;
    uint32_t debug_box_count = 0;
    uint32_t dig_edit_count = 0;
    uint32_t dig_removed_leaf_count = 0;
    uint32_t local_edit_depth = 0;
};

struct SurfaceNetMesh {
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<uint32_t> vertex_depths;
    std::vector<uint32_t> triangle_indices;
    float bounds_radius = 0.0f;
    uint32_t source_depth = 0;
    uint32_t occupied_voxel_count = 0;
    uint32_t candidate_cube_count = 0;
    uint32_t material_id = 5;
    uint32_t dig_edit_count = 0;
    uint32_t local_edit_depth = 0;
    uint32_t local_patch_count = 0;
    uint32_t local_patch_depth = 0;
    uint32_t local_vertex_count = 0;
    uint32_t local_triangle_count = 0;
};

struct VoxelOccupancyCache {
    std::vector<VoxelKey> leaf_keys;
    float bounds_radius = 0.0f;
    uint32_t depth = 0;

    bool empty() const { return leaf_keys.empty(); }
    void clear() {
        leaf_keys.clear();
        bounds_radius = 0.0f;
        depth = 0;
    }
};

struct QuantizedMesh {
    std::vector<QuantizedMeshVertex> vertices;
    std::vector<uint32_t> triangle_indices;
    std::vector<uint32_t> line_indices;
    std::vector<uint32_t> stitch_triangle_indices;
    std::vector<uint32_t> stitch_line_indices;
    SparseVoxelOctree svo;
    SurfaceNetMesh surface_net;
    SurfaceNetMesh surface_net_base_cache;
    VoxelOccupancyCache voxel_occupancy_cache;
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
    const PointCloud& points,
    const MarchingCubesConfig& config = {}
);

QuantizedMesh rebuild_quantized_mesh_voxels(
    QuantizedMesh mesh,
    const MarchingCubesConfig& config = {}
);

QuantizedMeshValidation validate_quantized_mesh(const QuantizedMesh& mesh);

} // namespace ae
