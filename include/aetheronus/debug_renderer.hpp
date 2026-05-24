#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/meshing.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/spaceship.hpp"
#include "aetheronus/topology.hpp"

#include <cstdint>
#include <vector>

namespace ae {

struct FreeCamera {
    Vec3 position = {1.13f, 0.84f, 3.09f};
    float yaw = 3.4916f;
    float pitch = -0.25f;
    float move_speed = 2.2f;
};

struct CameraView {
    Vec3 eye = {1.13f, 0.84f, 3.09f};
    Vec3 target = {0.0f, 0.0f, 0.0f};
    Vec3 up = {0.0f, 1.0f, 0.0f};
};

struct DebugRenderOptions {
    bool show_goldberg_grid = false;
    bool show_mesh_wire = false;
    bool show_points = false;
    bool follow_ship = false;
};

class DebugRenderer {
public:
    DebugRenderer() = default;
    ~DebugRenderer();

    DebugRenderer(const DebugRenderer&) = delete;
    DebugRenderer& operator=(const DebugRenderer&) = delete;

    bool initialize(const GoldbergTopology& topology, const std::vector<PointSample>& points, const QuantizedMesh& mesh);
    void update_mesh(const QuantizedMesh& mesh);
    void resize(int width, int height);
    void render(const CameraView& view, const SpaceshipState& ship, const DebugRenderOptions& options, bool show_fps, float fps);
    void shutdown();

private:
    void release_mesh_buffers();
    void render_fps_overlay(float fps);

    uint32_t shader_ = 0;
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
    uint32_t overlay_vao_ = 0;
    uint32_t overlay_vbo_ = 0;
    int line_vertex_count_ = 0;
    int point_vertex_count_ = 0;
    int mesh_triangle_index_count_ = 0;
    int mesh_line_index_count_ = 0;
    int stitch_triangle_index_count_ = 0;
    int stitch_line_index_count_ = 0;
    int grid_ribbon_vertex_count_ = 0;
    int grid_ribbon_line_vertex_count_ = 0;
    int width_ = 960;
    int height_ = 540;
};

} // namespace ae
