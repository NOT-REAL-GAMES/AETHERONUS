#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/meshing.hpp"
#include "aetheronus/planet_scale.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/spaceship.hpp"
#include "aetheronus/topology.hpp"

#include <cstdint>
#include <vector>

namespace ae {

struct FreeCamera {
    Vec3 target = {0.0f, 0.0f, 0.0f};
    float yaw = 3.4916f;
    float pitch = -0.25f;
    float distance = PlanetRadiusKilometers * 3.4f;
    float move_speed = 1200.0f;
};

struct CameraView {
    Vec3 eye = {PlanetRadiusKilometers * 1.13f, PlanetRadiusKilometers * 0.84f, PlanetRadiusKilometers * 3.09f};
    Vec3 target = {0.0f, 0.0f, 0.0f};
    Vec3 up = {0.0f, 1.0f, 0.0f};
};

struct DebugRenderOptions {
    bool show_goldberg_grid = false;
    bool show_mesh_wire = false;
    bool show_points = false;
    bool show_cave_anchors = true;
    bool show_voxels = false;
    bool show_surface_net = false;
    bool follow_ship = false;
};

struct RendererPerfStats {
    double upload_ms = 0.0;
    double render_cpu_ms = 0.0;
    double gpu_mesh_ms = 0.0;
    double gpu_surface_net_ms = 0.0;
    double gpu_debug_ms = 0.0;
    double voxel_compute_dispatch_ms = 0.0;
    uint64_t mesh_upload_bytes = 0;
    uint64_t surface_net_upload_bytes = 0;
    uint64_t voxel_dynamic_upload_bytes = 0;
    uint32_t mesh_vertices = 0;
    uint32_t mesh_triangles = 0;
    uint32_t stitch_triangles = 0;
    uint32_t mesh_lines = 0;
    uint32_t stitch_lines = 0;
    uint32_t surface_net_vertices = 0;
    uint32_t surface_net_triangles = 0;
    uint32_t cave_interior_vertices = 0;
    uint32_t cave_interior_triangles = 0;
    uint32_t voxel_debug_lines = 0;
    uint32_t draw_calls = 0;
    bool voxel_compute_available = false;
    bool voxel_compute_used = false;
    uint32_t voxel_compute_fallback_code = 0;
};

class DebugRenderer {
public:
    DebugRenderer() = default;
    ~DebugRenderer();

    DebugRenderer(const DebugRenderer&) = delete;
    DebugRenderer& operator=(const DebugRenderer&) = delete;

    bool initialize(const GoldbergTopology& topology, const PointCloud& points, const QuantizedMesh& mesh);
    void update_mesh(const QuantizedMesh& mesh);
    void update_cave_interiors(const SurfaceNetMesh& cave_interiors);
    void update_terrain_holes(const std::vector<LocalVoxelFeature>& holes);
    void update_cave_dig_transitions(const std::vector<LocalVoxelFeature>& active_holes, const VoxelEditSet& edits);
    void update_terrain_height_masks(const std::vector<TerrainHeightMask>& masks);
    void resize(int width, int height);
    void render(const CameraView& view, const SpaceshipState& ship, const DebugRenderOptions& options, bool show_fps, float fps);
    void render_progress_overlay(double progress);
    void shutdown();
    const RendererPerfStats& perf_stats() const { return perf_stats_; }

private:
    void release_mesh_buffers();
    void render_fps_overlay(float fps);
    void begin_gpu_timer(uint32_t query);
    void end_gpu_timer();
    void read_gpu_timer(uint32_t query, double& milliseconds);
    bool initialize_voxel_compute();
    void upload_voxel_compute_buffers(const SparseVoxelOctree& svo);
    bool render_voxel_compute(const CameraView& view, float aspect, float far_plane);
    bool initialize_cave_anchor_compute();
    void upload_cave_anchor_compute_buffers(const QuantizedMesh& mesh);
    bool render_cave_anchor_compute(const CameraView& view, float aspect, float far_plane);
    void rebuild_terrain_transition_buffer();

    uint32_t shader_ = 0;
    uint32_t surface_net_shader_ = 0;
    uint32_t voxel_compute_shader_ = 0;
    uint32_t cave_anchor_compute_shader_ = 0;
    uint32_t line_vao_ = 0;
    uint32_t line_vbo_ = 0;
    uint32_t point_vao_ = 0;
    uint32_t point_vbo_ = 0;
    uint32_t mesh_vao_ = 0;
    uint32_t mesh_vbo_ = 0;
    uint32_t mesh_triangle_ebo_ = 0;
    uint32_t mesh_line_ebo_ = 0;
    uint32_t stitch_triangle_ebo_ = 0;
    uint32_t stitch_line_ebo_ = 0;
    uint32_t grid_ribbon_vao_ = 0;
    uint32_t grid_ribbon_vbo_ = 0;
    uint32_t grid_ribbon_line_vao_ = 0;
    uint32_t grid_ribbon_line_vbo_ = 0;
    uint32_t voxel_vao_ = 0;
    uint32_t voxel_vbo_ = 0;
    uint32_t voxel_node_ssbo_ = 0;
    uint32_t voxel_counter_ssbo_ = 0;
    uint32_t voxel_indirect_buffer_ = 0;
    uint32_t cave_anchor_ssbo_ = 0;
    uint32_t cave_anchor_counter_ssbo_ = 0;
    uint32_t cave_anchor_indirect_buffer_ = 0;
    uint32_t cave_anchor_vao_ = 0;
    uint32_t cave_anchor_vbo_ = 0;
    uint32_t surface_net_vao_ = 0;
    uint32_t surface_net_vbo_ = 0;
    uint32_t surface_net_ebo_ = 0;
    uint32_t cave_interior_vao_ = 0;
    uint32_t cave_interior_vbo_ = 0;
    uint32_t cave_interior_ebo_ = 0;
    uint32_t cave_transition_vao_ = 0;
    uint32_t cave_transition_vbo_ = 0;
    uint32_t terrain_mask_texture_ = 0;
    uint32_t overlay_vao_ = 0;
    uint32_t overlay_vbo_ = 0;
    uint32_t gpu_query_mesh_ = 0;
    uint32_t gpu_query_surface_net_ = 0;
    uint32_t gpu_query_debug_ = 0;
    int shader_mvp_location_ = -1;
    int shader_point_size_location_ = -1;
    int shader_point_style_location_ = -1;
    int shader_terrain_hole_count_location_ = -1;
    int shader_terrain_holes_location_ = -1;
    int shader_terrain_transition_tangent_location_ = -1;
    int shader_terrain_transition_bitangent_seed_location_ = -1;
    int shader_terrain_transition_shape_location_ = -1;
    int shader_cave_dig_count_location_ = -1;
    int shader_cave_dig_spheres_location_ = -1;
    int shader_cave_dig_planes_location_ = -1;
    int shader_terrain_mask_count_location_ = -1;
    int shader_terrain_mask_sampler_location_ = -1;
    int shader_terrain_mask_center_radius_location_ = -1;
    int shader_terrain_mask_tangent_location_ = -1;
    int shader_terrain_mask_bitangent_location_ = -1;
    int surface_mvp_location_ = -1;
    int surface_camera_location_ = -1;
    int surface_light_location_ = -1;
    uint32_t gpu_timer_frame_ = 0;
    int line_vertex_count_ = 0;
    int point_vertex_count_ = 0;
    int mesh_triangle_index_count_ = 0;
    int mesh_line_index_count_ = 0;
    int stitch_triangle_index_count_ = 0;
    int stitch_line_index_count_ = 0;
    int grid_ribbon_vertex_count_ = 0;
    int grid_ribbon_line_vertex_count_ = 0;
    int voxel_line_vertex_count_ = 0;
    int cave_anchor_vertex_count_ = 0;
    int surface_net_index_count_ = 0;
    bool surface_net_auto_visible_ = false;
    int cave_interior_index_count_ = 0;
    int cave_transition_vertex_count_ = 0;
    int width_ = 960;
    int height_ = 540;
    uint32_t overlay_buffer_capacity_bytes_ = 0;
    uint32_t voxel_compute_node_count_ = 0;
    uint32_t voxel_compute_max_boxes_ = 0;
    uint32_t voxel_compute_vertex_capacity_ = 0;
    bool voxel_compute_available_ = false;
    bool voxel_compute_enabled_ = false;
    bool cave_anchor_compute_available_ = false;
    bool cave_anchor_compute_enabled_ = false;
    uint32_t cave_anchor_compute_count_ = 0;
    uint32_t cave_anchor_compute_capacity_ = 0;
    int voxel_compute_node_count_location_ = -1;
    int voxel_compute_bounds_radius_location_ = -1;
    int voxel_compute_svo_depth_location_ = -1;
    int voxel_compute_max_boxes_location_ = -1;
    int voxel_compute_planet_radius_location_ = -1;
    int voxel_compute_eye_location_ = -1;
    int voxel_compute_range_center_location_ = -1;
    int voxel_compute_forward_location_ = -1;
    int voxel_compute_right_location_ = -1;
    int voxel_compute_up_location_ = -1;
    int voxel_compute_tan_half_fov_y_location_ = -1;
    int voxel_compute_tan_half_fov_x_location_ = -1;
    int voxel_compute_near_plane_location_ = -1;
    int voxel_compute_far_plane_location_ = -1;
    int voxel_compute_occlusion_radius_location_ = -1;
    int voxel_compute_range_radius_location_ = -1;
    int voxel_compute_debug_box_size_location_ = -1;
    int cave_anchor_compute_anchor_count_location_ = -1;
    int cave_anchor_compute_max_visible_location_ = -1;
    int cave_anchor_compute_planet_radius_location_ = -1;
    int cave_anchor_compute_eye_location_ = -1;
    int cave_anchor_compute_forward_location_ = -1;
    int cave_anchor_compute_right_location_ = -1;
    int cave_anchor_compute_up_location_ = -1;
    int cave_anchor_compute_tan_half_fov_y_location_ = -1;
    int cave_anchor_compute_tan_half_fov_x_location_ = -1;
    int cave_anchor_compute_near_plane_location_ = -1;
    int cave_anchor_compute_far_plane_location_ = -1;
    int cave_anchor_compute_occlusion_radius_location_ = -1;
    int cave_anchor_compute_max_distance_location_ = -1;
    struct CaveDigTransition {
        Vec3 center_mesh;
        float radius_km = 0.0f;
        Vec3 normal_mesh;
        float boundary_dot_km = 0.0f;
    };
    SparseVoxelOctree current_svo_;
    std::vector<LocalVoxelFeature> terrain_holes_;
    std::vector<CaveDigTransition> cave_dig_transitions_;
    std::vector<TerrainHeightMask> terrain_masks_;
    uint32_t terrain_mask_resolution_ = 0;
    uint32_t terrain_mask_layer_count_ = 0;
    RendererPerfStats perf_stats_;
};

} // namespace ae
