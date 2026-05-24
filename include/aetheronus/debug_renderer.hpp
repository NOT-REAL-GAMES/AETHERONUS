#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/meshing.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/topology.hpp"

#include <cstdint>
#include <vector>

namespace ae {

struct OrbitCamera {
    float yaw = 0.35f;
    float pitch = 0.25f;
    float distance = 3.4f;
};

struct DebugRenderOptions {
    bool show_goldberg_grid = false;
    bool show_mesh_wire = false;
    bool show_points = false;
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
    void render(const OrbitCamera& camera, const DebugRenderOptions& options, bool show_fps, float fps);
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
