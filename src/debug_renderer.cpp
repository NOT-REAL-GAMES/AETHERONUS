#include "aetheronus/debug_renderer.hpp"

#include "aetheronus/planet_scale.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace ae {
namespace {

struct DebugVertex {
    Vec3 position;
    Vec3 color;
};

struct SurfaceNetVertex {
    Vec3 position;
    Vec3 normal;
    Vec3 color;
};

struct VoxelDebugView {
    Vec3 eye;
    Vec3 range_center;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float tan_half_fov_y = 1.0f;
    float tan_half_fov_x = 1.0f;
    float near_plane = 0.05f;
    float far_plane = 1.0f;
    float occlusion_radius = PlanetRadiusKilometers;
    float range_radius = 100.0f;
    float debug_box_size = 2.0f;
};

Vec3 planet_to_world(Vec3 position) {
    return position * PlanetRadiusKilometers;
}

Vec3 camera_surface_focus(Vec3 eye) {
    if (length(eye) <= 0.0001f) {
        return {0.0f, PlanetRadiusKilometers, 0.0f};
    }
    return normalize(eye) * PlanetRadiusKilometers;
}

const char* VertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_color;

uniform mat4 u_mvp;
uniform float u_point_size;

out vec3 v_color;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    gl_PointSize = u_point_size;
    v_color = a_color;
}
)glsl";

const char* FragmentShaderSource = R"glsl(
#version 330 core
in vec3 v_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
)glsl";

const char* SurfaceNetVertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec3 a_color;

uniform mat4 u_mvp;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_position = a_position;
    v_normal = normalize(a_normal);
    v_color = a_color;
}
)glsl";

const char* SurfaceNetFragmentShaderSource = R"glsl(
#version 330 core
in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;

uniform vec3 u_camera_position;
uniform vec3 u_light_direction;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(-u_light_direction);
    vec3 v = normalize(u_camera_position - v_position);
    vec3 h = normalize(l + v);

    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(n, h), 0.0), 48.0) * 0.34;
    vec3 ambient = v_color * 0.18;
    vec3 lit = ambient + v_color * diffuse * 0.82 + vec3(1.0, 0.90, 0.68) * specular;
    frag_color = vec4(lit, 1.0);
}
)glsl";

uint32_t compile_shader(uint32_t type, const char* source) {
    const uint32_t shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048] = {};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compile failed: " << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

uint32_t build_shader_program_from_sources(const char* vertex_source, const char* fragment_source) {
    const uint32_t vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    const uint32_t fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (vertex == 0 || fragment == 0) {
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        return 0;
    }

    const uint32_t program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    int ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = {};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Shader link failed: " << log << std::endl;
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

uint32_t build_shader_program() {
    return build_shader_program_from_sources(VertexShaderSource, FragmentShaderSource);
}

uint32_t build_surface_net_shader_program() {
    return build_shader_program_from_sources(SurfaceNetVertexShaderSource, SurfaceNetFragmentShaderSource);
}

Vec3 point_color(uint32_t source_cell_id, uint32_t owner_cell_id, uint32_t material_id) {
    if (source_cell_id != owner_cell_id) {
        return {1.0f, 0.12f, 0.22f};
    }

    switch (material_id) {
        case 1:
            return {1.0f, 0.58f, 0.18f};
        case 2:
            return {0.45f, 0.84f, 1.0f};
        case 3:
            return {1.0f, 0.72f, 0.28f};
        default:
            return {0.66f, 0.92f, 0.92f};
    }
}

void append_goldberg_cell_ring_band(
    std::vector<DebugVertex>& triangles,
    std::vector<DebugVertex>& lines,
    std::vector<DebugVertex>& outline_lines,
    const GoldbergTopology& topology,
    const GoldbergCell& cell
) {
    if (cell.corner_indices.size() < 3) {
        return;
    }

    constexpr float RibbonRadius = 1.018f;
    constexpr float InnerInset = 0.07f;

    const bool pentagon = cell.kind == GoldbergCellKind::Pentagon;
    const Vec3 fill_color = pentagon
        ? Vec3{0.22f, 0.12f, 0.035f}
        : Vec3{0.08f, 0.20f, 0.035f};
    const Vec3 line_color = pentagon
        ? Vec3{1.0f, 0.66f, 0.18f}
        : Vec3{0.56f, 0.90f, 0.22f};
    const Vec3 outline_color = pentagon
        ? Vec3{0.80f, 0.42f, 0.10f}
        : Vec3{0.14f, 0.42f, 0.50f};

    std::vector<Vec3> outer_loop;
    std::vector<Vec3> inner_loop;
    outer_loop.reserve(cell.corner_indices.size());
    inner_loop.reserve(cell.corner_indices.size());
    for (uint32_t corner_index : cell.corner_indices) {
        if (corner_index >= topology.vertices.size()) {
            continue;
        }
        const Vec3 corner = topology.vertices[corner_index].position;
        outer_loop.push_back(planet_to_world(corner * RibbonRadius));
        inner_loop.push_back(planet_to_world(normalize(lerp(corner, cell.center, InnerInset)) * RibbonRadius));
    }

    if (outer_loop.size() < 3 || inner_loop.size() != outer_loop.size()) {
        return;
    }

    for (uint32_t i = 0; i < outer_loop.size(); ++i) {
        const uint32_t next = (i + 1) % static_cast<uint32_t>(outer_loop.size());
        const Vec3 outer0 = outer_loop[i];
        const Vec3 outer1 = outer_loop[next];
        const Vec3 inner0 = inner_loop[i];
        const Vec3 inner1 = inner_loop[next];

        triangles.push_back({outer0, fill_color});
        triangles.push_back({outer1, fill_color});
        triangles.push_back({inner1, fill_color});
        triangles.push_back({outer0, fill_color});
        triangles.push_back({inner1, fill_color});
        triangles.push_back({inner0, fill_color});

        lines.push_back({outer0, line_color});
        lines.push_back({outer1, line_color});
        lines.push_back({inner0, line_color});
        lines.push_back({inner1, line_color});
        lines.push_back({outer0, line_color});
        lines.push_back({inner0, line_color});

        outline_lines.push_back({outer0, outline_color});
        outline_lines.push_back({outer1, outline_color});
    }
}

void append_voxel_wire_cube(std::vector<DebugVertex>& vertices, Vec3 center, float size) {
    const Vec3 color = {0.95f, 0.34f, 0.96f};
    const float h = size * 0.48f;
    const std::array<Vec3, 8> corners = {{
        {center.x - h, center.y - h, center.z - h},
        {center.x + h, center.y - h, center.z - h},
        {center.x + h, center.y + h, center.z - h},
        {center.x - h, center.y + h, center.z - h},
        {center.x - h, center.y - h, center.z + h},
        {center.x + h, center.y - h, center.z + h},
        {center.x + h, center.y + h, center.z + h},
        {center.x - h, center.y + h, center.z + h},
    }};
    constexpr std::array<std::array<uint32_t, 2>, 12> edges = {{
        {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
        {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
        {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
    }};
    for (const auto& edge : edges) {
        vertices.push_back({corners[edge[0]], color});
        vertices.push_back({corners[edge[1]], color});
    }
}

Vec3 svo_node_center(const SparseVoxelOctree& svo, const SparseVoxelOctreeNode& node) {
    const float cell_size = (svo.bounds_radius * 2.0f) / static_cast<float>(1u << svo.depth);
    return {
        -svo.bounds_radius + (static_cast<float>(node.origin_x) + static_cast<float>(node.size) * 0.5f) * cell_size,
        -svo.bounds_radius + (static_cast<float>(node.origin_y) + static_cast<float>(node.size) * 0.5f) * cell_size,
        -svo.bounds_radius + (static_cast<float>(node.origin_z) + static_cast<float>(node.size) * 0.5f) * cell_size,
    };
}

float svo_node_world_size(const SparseVoxelOctree& svo, const SparseVoxelOctreeNode& node) {
    const float cell_size = (svo.bounds_radius * 2.0f) / static_cast<float>(1u << svo.depth);
    return static_cast<float>(node.size) * cell_size;
}

bool voxel_box_visible(Vec3 center, float radius, const VoxelDebugView& view) {
    if (length(center - view.range_center) > view.range_radius + radius) {
        return false;
    }

    const Vec3 relative = center - view.eye;
    const float z = dot(relative, view.forward);
    if (z + radius < view.near_plane || z - radius > view.far_plane) {
        return false;
    }

    const float x = dot(relative, view.right);
    const float y = dot(relative, view.up);
    const float perspective_z = std::max(z, view.near_plane);
    if (std::fabs(x) - radius > perspective_z * view.tan_half_fov_x ||
        std::fabs(y) - radius > perspective_z * view.tan_half_fov_y) {
        return false;
    }

    const float eye_radius = length(view.eye);
    if (eye_radius > view.occlusion_radius + 1.0f) {
        const float horizon_limit = view.occlusion_radius * view.occlusion_radius;
        const float box_margin = radius * eye_radius;
        if (dot(center, view.eye) + box_margin < horizon_limit) {
            return false;
        }
    }

    return true;
}

void append_subdivided_voxel_wire_cubes(
    std::vector<DebugVertex>& vertices,
    Vec3 center,
    float size,
    const VoxelDebugView& view,
    uint32_t& emitted_boxes,
    uint32_t max_boxes
) {
    const float target_size = std::max(0.25f, view.debug_box_size);
    const uint32_t steps = std::max(1u, static_cast<uint32_t>(std::ceil(size / target_size)));
    const float box_size = size / static_cast<float>(steps);
    const float box_radius = box_size * 0.8661f;
    const Vec3 min_corner = center - Vec3{size * 0.5f, size * 0.5f, size * 0.5f};
    const Vec3 range_min = view.range_center - Vec3{view.range_radius + box_radius, view.range_radius + box_radius, view.range_radius + box_radius};
    const Vec3 range_max = view.range_center + Vec3{view.range_radius + box_radius, view.range_radius + box_radius, view.range_radius + box_radius};

    auto first_step = [&](float minimum, float range_minimum) {
        return std::clamp(static_cast<int32_t>(std::floor((range_minimum - minimum) / box_size)), 0, static_cast<int32_t>(steps - 1u));
    };
    auto last_step = [&](float minimum, float range_maximum) {
        return std::clamp(static_cast<int32_t>(std::ceil((range_maximum - minimum) / box_size)), 0, static_cast<int32_t>(steps - 1u));
    };

    const int32_t ix0 = first_step(min_corner.x, range_min.x);
    const int32_t iy0 = first_step(min_corner.y, range_min.y);
    const int32_t iz0 = first_step(min_corner.z, range_min.z);
    const int32_t ix1 = last_step(min_corner.x, range_max.x);
    const int32_t iy1 = last_step(min_corner.y, range_max.y);
    const int32_t iz1 = last_step(min_corner.z, range_max.z);

    for (int32_t z = iz0; z <= iz1 && emitted_boxes < max_boxes; ++z) {
        for (int32_t y = iy0; y <= iy1 && emitted_boxes < max_boxes; ++y) {
            for (int32_t x = ix0; x <= ix1 && emitted_boxes < max_boxes; ++x) {
                const Vec3 box_center = {
                    min_corner.x + (static_cast<float>(x) + 0.5f) * box_size,
                    min_corner.y + (static_cast<float>(y) + 0.5f) * box_size,
                    min_corner.z + (static_cast<float>(z) + 0.5f) * box_size,
                };
                if (!voxel_box_visible(box_center, box_radius, view)) {
                    continue;
                }
                append_voxel_wire_cube(vertices, box_center, box_size);
                ++emitted_boxes;
            }
        }
    }
}

void append_svo_debug_boxes(
    std::vector<DebugVertex>& vertices,
    const SparseVoxelOctree& svo,
    uint32_t node_index,
    const VoxelDebugView& view,
    uint32_t& emitted_boxes
) {
    if (node_index >= svo.nodes.size() || emitted_boxes >= svo.debug_max_boxes) {
        return;
    }

    const SparseVoxelOctreeNode& node = svo.nodes[node_index];
    const Vec3 center = planet_to_world(svo_node_center(svo, node));
    const float size = svo_node_world_size(svo, node) * PlanetRadiusKilometers;
    const float radius = size * 0.8661f;
    if (!voxel_box_visible(center, radius, view)) {
        return;
    }

    if (node.child_mask == 0u || node.depth >= svo.debug_draw_depth) {
        if (size > view.debug_box_size * 1.5f) {
            append_subdivided_voxel_wire_cubes(vertices, center, size, view, emitted_boxes, svo.debug_max_boxes);
        } else {
            append_voxel_wire_cube(vertices, center, size);
            ++emitted_boxes;
        }
        return;
    }

    struct ChildCandidate {
        uint32_t index = 0;
        float distance2 = 0.0f;
    };
    std::array<ChildCandidate, 8> children = {};
    uint32_t child_count = 0;
    uint32_t child_slot = 0;
    for (uint32_t child = 0; child < 8; ++child) {
        if ((node.child_mask & (1u << child)) == 0u) {
            continue;
        }
        const uint32_t child_index = node.first_child + child_slot;
        const Vec3 child_center = planet_to_world(svo_node_center(svo, svo.nodes[child_index]));
        const Vec3 to_child = child_center - view.range_center;
        children[child_count++] = {child_index, dot(to_child, to_child)};
        ++child_slot;
    }

    std::sort(children.begin(), children.begin() + child_count, [](const ChildCandidate& lhs, const ChildCandidate& rhs) {
        return lhs.distance2 < rhs.distance2;
    });
    for (uint32_t i = 0; i < child_count && emitted_boxes < svo.debug_max_boxes; ++i) {
        append_svo_debug_boxes(vertices, svo, children[i].index, view, emitted_boxes);
    }
}

std::vector<DebugVertex> build_visible_voxel_vertices(
    const SparseVoxelOctree& svo,
    const CameraView& camera,
    float aspect,
    float far_plane
) {
    std::vector<DebugVertex> vertices;
    vertices.reserve(static_cast<size_t>(std::min(svo.debug_box_count, 65536u)) * 24u);
    if (svo.nodes.empty() || svo.bounds_radius <= 0.0f || svo.depth == 0u) {
        return vertices;
    }

    constexpr float FovY = 50.0f * Pi / 180.0f;
    const Vec3 forward = normalize(camera.target - camera.eye);
    const Vec3 right = normalize(cross(forward, camera.up));
    const Vec3 up = cross(right, forward);
    VoxelDebugView view;
    view.eye = camera.eye;
    view.range_center = camera_surface_focus(camera.eye);
    view.forward = forward;
    view.right = right;
    view.up = up;
    view.tan_half_fov_y = std::tan(FovY * 0.5f);
    view.tan_half_fov_x = view.tan_half_fov_y * aspect;
    view.far_plane = far_plane;
    view.occlusion_radius = PlanetRadiusKilometers * 0.995f;
    view.range_radius = 100.0f;
    view.debug_box_size = 2.0f;

    uint32_t emitted_boxes = 0;
    append_svo_debug_boxes(vertices, svo, 0, view, emitted_boxes);
    return vertices;
}

void upload_vertex_buffer(uint32_t& vao, uint32_t& vbo, const std::vector<DebugVertex>& vertices) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(DebugVertex)), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(sizeof(Vec3)));
    glBindVertexArray(0);
}

void upload_surface_net_buffer(
    uint32_t& vao,
    uint32_t& vbo,
    uint32_t& ebo,
    const std::vector<SurfaceNetVertex>& vertices,
    const std::vector<uint32_t>& indices
) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(SurfaceNetVertex)), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceNetVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceNetVertex), reinterpret_cast<void*>(sizeof(Vec3)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceNetVertex), reinterpret_cast<void*>(sizeof(Vec3) * 2u));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(indices.size() * sizeof(uint32_t)),
        indices.data(),
        GL_STATIC_DRAW
    );
    glBindVertexArray(0);
}

void upload_indexed_mesh_buffer(
    uint32_t& vao,
    uint32_t& vbo,
    uint32_t& triangle_ebo,
    uint32_t& line_ebo,
    uint32_t& stitch_triangle_ebo,
    uint32_t& stitch_line_ebo,
    const std::vector<DebugVertex>& vertices,
    const std::vector<uint32_t>& triangle_indices,
    const std::vector<uint32_t>& line_indices,
    const std::vector<uint32_t>& stitch_triangle_indices,
    const std::vector<uint32_t>& stitch_line_indices
) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &triangle_ebo);
    glGenBuffers(1, &line_ebo);
    glGenBuffers(1, &stitch_triangle_ebo);
    glGenBuffers(1, &stitch_line_ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(DebugVertex)), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(sizeof(Vec3)));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(triangle_indices.size() * sizeof(uint32_t)),
        triangle_indices.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, line_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(line_indices.size() * sizeof(uint32_t)),
        line_indices.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, stitch_triangle_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(stitch_triangle_indices.size() * sizeof(uint32_t)),
        stitch_triangle_indices.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, stitch_line_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(stitch_line_indices.size() * sizeof(uint32_t)),
        stitch_line_indices.data(),
        GL_STATIC_DRAW
    );

    glBindVertexArray(0);
}

std::vector<DebugVertex> build_debug_mesh_vertices(const QuantizedMesh& mesh) {
    std::vector<DebugVertex> vertices;
    vertices.reserve(mesh.vertices.size());
    for (const QuantizedMeshVertex& vertex : mesh.vertices) {
        Vec3 color = vertex.material_id == 2u
            ? Vec3{0.045f, 0.095f, 0.045f}
            : vertex.material_id == 4u
            ? Vec3{0.010f, 0.014f, 0.018f}
            : vertex.material_id == 1u
            ? Vec3{0.20f, 0.15f, 0.10f}
            : Vec3{0.07f, 0.15f, 0.18f};
        if (vertex.fracture_chunk_id != 0u && vertex.material_id != 2u && vertex.material_id != 4u) {
            const uint32_t hash = vertex.fracture_chunk_id * 747796405u + 2891336453u;
            const float warm = static_cast<float>((hash >> 8) & 0xffu) / 255.0f;
            const float cool = static_cast<float>((hash >> 16) & 0xffu) / 255.0f;
            color = {
                0.070f + warm * 0.050f,
                0.145f + (1.0f - warm) * 0.035f,
                0.175f + cool * 0.040f,
            };
        }
        vertices.push_back({planet_to_world(vertex.position), color});
    }
    return vertices;
}

std::vector<SurfaceNetVertex> build_surface_net_vertices(const SurfaceNetMesh& surface_net) {
    std::vector<SurfaceNetVertex> vertices;
    vertices.reserve(surface_net.vertices.size());
    const Vec3 color = surface_net.material_id == 5u ? Vec3{0.95f, 0.62f, 0.18f} : Vec3{0.86f, 0.74f, 0.42f};
    for (uint32_t i = 0; i < surface_net.vertices.size(); ++i) {
        const Vec3 normal = i < surface_net.normals.size() ? normalize(surface_net.normals[i]) : normalize(surface_net.vertices[i]);
        vertices.push_back({planet_to_world(surface_net.vertices[i]), normal, color});
    }
    return vertices;
}

Vec3 ship_forward(const SpaceshipState& ship) {
    return normalize(ship.forward);
}

void append_ship_triangle(std::vector<DebugVertex>& vertices, Vec3 a, Vec3 b, Vec3 c, Vec3 color) {
    vertices.push_back({a, color});
    vertices.push_back({b, color});
    vertices.push_back({c, color});
}

std::vector<DebugVertex> build_spaceship_vertices(const SpaceshipState& ship) {
    const Vec3 forward = ship_forward(ship);
    Vec3 visual_up = ship.up - forward * dot(ship.up, forward);
    if (length(visual_up) <= 0.000001f) {
        visual_up = {0.0f, 1.0f, 0.0f};
    }
    visual_up = normalize(visual_up);
    const Vec3 right = normalize(cross(forward, visual_up));
    const Vec3 center = spaceship_position(ship);

    const Vec3 nose = center + forward * 0.080f;
    const Vec3 tail = center - forward * 0.072f;
    const Vec3 left = center - forward * 0.030f - right * 0.052f - visual_up * 0.004f;
    const Vec3 right_wing = center - forward * 0.030f + right * 0.052f - visual_up * 0.004f;
    const Vec3 top = center + forward * 0.008f + visual_up * 0.034f;
    const Vec3 keel = center - forward * 0.020f - visual_up * 0.020f;

    std::vector<DebugVertex> vertices;
    vertices.reserve(18);
    append_ship_triangle(vertices, nose, left, top, {0.82f, 0.92f, 1.0f});
    append_ship_triangle(vertices, right_wing, nose, top, {0.46f, 0.76f, 1.0f});
    append_ship_triangle(vertices, left, tail, top, {0.28f, 0.42f, 0.72f});
    append_ship_triangle(vertices, tail, right_wing, top, {0.22f, 0.36f, 0.66f});
    append_ship_triangle(vertices, nose, keel, left, {0.12f, 0.18f, 0.28f});
    append_ship_triangle(vertices, right_wing, keel, nose, {0.10f, 0.16f, 0.26f});
    return vertices;
}

std::vector<DebugVertex> build_spaceship_trail_vertices(const SpaceshipState& ship) {
    std::vector<DebugVertex> vertices;
    if (ship.trail_count < 2) {
        return vertices;
    }

    vertices.reserve((ship.trail_count - 1u) * 2u);
    const uint32_t first_index = (ship.trail_head + SpaceshipState::TrailCapacity - ship.trail_count) % SpaceshipState::TrailCapacity;
    auto trail_point = [&](uint32_t index) {
        return ship.trail[(first_index + index) % SpaceshipState::TrailCapacity];
    };
    for (uint32_t i = 1; i < ship.trail_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(ship.trail_count - 1u);
        const Vec3 color = {0.16f + t * 0.42f, 0.38f + t * 0.42f, 0.58f + t * 0.38f};
        vertices.push_back({trail_point(i - 1u), color});
        vertices.push_back({trail_point(i), color});
    }
    return vertices;
}

void append_line(std::vector<DebugVertex>& vertices, float x0, float y0, float x1, float y1, Vec3 color) {
    vertices.push_back({{x0, y0, 0.0f}, color});
    vertices.push_back({{x1, y1, 0.0f}, color});
}

void append_seven_segment(std::vector<DebugVertex>& vertices, int digit, float x, float y, float size, Vec3 color) {
    constexpr uint8_t masks[10] = {
        0b0111111, // 0
        0b0000110, // 1
        0b1011011, // 2
        0b1001111, // 3
        0b1100110, // 4
        0b1101101, // 5
        0b1111101, // 6
        0b0000111, // 7
        0b1111111, // 8
        0b1101111, // 9
    };

    const uint8_t mask = masks[digit];
    const float w = size;
    const float h = size * 1.7f;
    const float mid = y - h * 0.5f;
    const float bottom = y - h;

    if ((mask & 0b0000001) != 0) append_line(vertices, x, y, x + w, y, color);
    if ((mask & 0b0000010) != 0) append_line(vertices, x + w, y, x + w, mid, color);
    if ((mask & 0b0000100) != 0) append_line(vertices, x + w, mid, x + w, bottom, color);
    if ((mask & 0b0001000) != 0) append_line(vertices, x, bottom, x + w, bottom, color);
    if ((mask & 0b0010000) != 0) append_line(vertices, x, mid, x, bottom, color);
    if ((mask & 0b0100000) != 0) append_line(vertices, x, y, x, mid, color);
    if ((mask & 0b1000000) != 0) append_line(vertices, x, mid, x + w, mid, color);
}

void append_letter(std::vector<DebugVertex>& vertices, char letter, float x, float y, float size, Vec3 color) {
    const float w = size;
    const float h = size * 1.7f;
    const float mid = y - h * 0.5f;
    const float bottom = y - h;

    switch (letter) {
        case 'F':
            append_line(vertices, x, y, x + w, y, color);
            append_line(vertices, x, y, x, bottom, color);
            append_line(vertices, x, mid, x + w * 0.85f, mid, color);
            break;
        case 'P':
            append_line(vertices, x, y, x, bottom, color);
            append_line(vertices, x, y, x + w, y, color);
            append_line(vertices, x + w, y, x + w, mid, color);
            append_line(vertices, x, mid, x + w, mid, color);
            break;
        case 'S':
            append_seven_segment(vertices, 5, x, y, size, color);
            break;
        default:
            break;
    }
}

std::vector<DebugVertex> build_fps_overlay_vertices(float fps) {
    std::vector<DebugVertex> vertices;
    const Vec3 color = {0.72f, 1.0f, 0.78f};
    const float size = 0.043f;
    const float advance = 0.066f;
    const float y = 0.91f;
    float x = -0.94f;

    append_letter(vertices, 'F', x, y, size, color);
    x += advance;
    append_letter(vertices, 'P', x, y, size, color);
    x += advance;
    append_letter(vertices, 'S', x, y, size, color);
    x += advance * 1.25f;

    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%05d", static_cast<int>(std::round(std::clamp(fps, 0.0f, 99999.0f))));
    for (const char* cursor = buffer; *cursor != '\0'; ++cursor) {
        append_seven_segment(vertices, *cursor - '0', x, y, size, color);
        x += advance;
    }

    return vertices;
}

} // namespace

DebugRenderer::~DebugRenderer() {
    shutdown();
}

bool DebugRenderer::initialize(const GoldbergTopology& topology, const PointCloud& points, const QuantizedMesh& mesh) {
    shutdown();

    shader_ = build_shader_program();
    if (shader_ == 0) {
        return false;
    }
    surface_net_shader_ = build_surface_net_shader_program();
    if (surface_net_shader_ == 0) {
        return false;
    }

    std::vector<DebugVertex> line_vertices;
    std::vector<DebugVertex> grid_ribbon_vertices;
    std::vector<DebugVertex> grid_ribbon_line_vertices;
    line_vertices.reserve(topology.edge_count * 4);
    grid_ribbon_vertices.reserve(topology.edge_count * 12);
    grid_ribbon_line_vertices.reserve(topology.edge_count * 16);
    for (const GoldbergCell& cell : topology.cells) {
        append_goldberg_cell_ring_band(grid_ribbon_vertices, grid_ribbon_line_vertices, line_vertices, topology, cell);
    }

    std::vector<DebugVertex> point_vertices;
    point_vertices.reserve(points.size());
    for (uint32_t i = 0; i < points.size(); ++i) {
        point_vertices.push_back({
            planet_to_world(points.positions[i] * 1.018f),
            point_color(points.source_cell_ids[i], points.owner_cell_ids[i], points.material_ids[i]),
        });
    }

    update_mesh(mesh);
    upload_vertex_buffer(line_vao_, line_vbo_, line_vertices);
    upload_vertex_buffer(grid_ribbon_vao_, grid_ribbon_vbo_, grid_ribbon_vertices);
    upload_vertex_buffer(grid_ribbon_line_vao_, grid_ribbon_line_vbo_, grid_ribbon_line_vertices);
    upload_vertex_buffer(point_vao_, point_vbo_, point_vertices);
    glGenVertexArrays(1, &overlay_vao_);
    glGenBuffers(1, &overlay_vbo_);
    glBindVertexArray(overlay_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo_);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(sizeof(Vec3)));
    glBindVertexArray(0);

    line_vertex_count_ = static_cast<int>(line_vertices.size());
    point_vertex_count_ = static_cast<int>(point_vertices.size());
    grid_ribbon_vertex_count_ = static_cast<int>(grid_ribbon_vertices.size());
    grid_ribbon_line_vertex_count_ = static_cast<int>(grid_ribbon_line_vertices.size());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.015f, 0.018f, 0.024f, 1.0f);
    return true;
}

void DebugRenderer::update_mesh(const QuantizedMesh& mesh) {
    release_mesh_buffers();

    const std::vector<DebugVertex> mesh_vertices = build_debug_mesh_vertices(mesh);
    const std::vector<DebugVertex> voxel_vertices;
    const std::vector<SurfaceNetVertex> surface_net_vertices = build_surface_net_vertices(mesh.surface_net);
    current_svo_ = mesh.svo;
    upload_indexed_mesh_buffer(
        mesh_vao_,
        mesh_vbo_,
        mesh_triangle_ebo_,
        mesh_line_ebo_,
        stitch_triangle_ebo_,
        stitch_line_ebo_,
        mesh_vertices,
        mesh.triangle_indices,
        mesh.line_indices,
        mesh.stitch_triangle_indices,
        mesh.stitch_line_indices
    );
    upload_vertex_buffer(voxel_vao_, voxel_vbo_, voxel_vertices);
    upload_surface_net_buffer(
        surface_net_vao_,
        surface_net_vbo_,
        surface_net_ebo_,
        surface_net_vertices,
        mesh.surface_net.triangle_indices
    );

    mesh_triangle_index_count_ = static_cast<int>(mesh.triangle_indices.size());
    mesh_line_index_count_ = static_cast<int>(mesh.line_indices.size());
    stitch_triangle_index_count_ = static_cast<int>(mesh.stitch_triangle_indices.size());
    stitch_line_index_count_ = static_cast<int>(mesh.stitch_line_indices.size());
    voxel_line_vertex_count_ = 0;
    surface_net_index_count_ = static_cast<int>(mesh.surface_net.triangle_indices.size());
}

void DebugRenderer::resize(int width, int height) {
    width_ = width > 0 ? width : 1;
    height_ = height > 0 ? height : 1;
}

void DebugRenderer::render(const CameraView& view, const SpaceshipState& ship, const DebugRenderOptions& options, bool show_fps, float fps) {
    glViewport(0, 0, width_, height_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float aspect = static_cast<float>(width_) / static_cast<float>(height_);
    const float far_plane = std::max(PlanetRadiusKilometers * 16.0f, length(view.eye) + PlanetRadiusKilometers * 4.0f);
    const Mat4 projection = perspective(50.0f * Pi / 180.0f, aspect, 0.05f, far_plane);
    const Mat4 view_matrix = look_at(view.eye, view.target, view.up);
    const Mat4 mvp = projection * view_matrix;

    glUseProgram(shader_);
    const int mvp_location = glGetUniformLocation(shader_, "u_mvp");
    const int point_size_location = glGetUniformLocation(shader_, "u_point_size");
    glUniformMatrix4fv(mvp_location, 1, GL_FALSE, mvp.m);

    glUniform1f(point_size_location, 1.0f);
    glBindVertexArray(mesh_vao_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_triangle_ebo_);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glDrawElements(GL_TRIANGLES, mesh_triangle_index_count_, GL_UNSIGNED_INT, nullptr);
    glDisable(GL_POLYGON_OFFSET_FILL);

    if (options.show_mesh_wire) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_line_ebo_);
        glDisableVertexAttribArray(1);
        glVertexAttrib3f(1, 0.42f, 0.92f, 1.0f);
        glDrawElements(GL_LINES, mesh_line_index_count_, GL_UNSIGNED_INT, nullptr);
        glEnableVertexAttribArray(1);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, stitch_triangle_ebo_);
    glDrawElements(GL_TRIANGLES, stitch_triangle_index_count_, GL_UNSIGNED_INT, nullptr);

    if (options.show_surface_net) {
        glUseProgram(surface_net_shader_);
        const int surface_mvp_location = glGetUniformLocation(surface_net_shader_, "u_mvp");
        const int surface_camera_location = glGetUniformLocation(surface_net_shader_, "u_camera_position");
        const int surface_light_location = glGetUniformLocation(surface_net_shader_, "u_light_direction");
        const Vec3 light_direction = normalize(Vec3{-0.35f, -0.75f, -0.50f});
        glUniformMatrix4fv(surface_mvp_location, 1, GL_FALSE, mvp.m);
        glUniform3f(surface_camera_location, view.eye.x, view.eye.y, view.eye.z);
        glUniform3f(surface_light_location, light_direction.x, light_direction.y, light_direction.z);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.0f, -1.0f);
        glBindVertexArray(surface_net_vao_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, surface_net_ebo_);
        glDrawElements(GL_TRIANGLES, surface_net_index_count_, GL_UNSIGNED_INT, nullptr);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glUseProgram(shader_);
        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, mvp.m);
        glUniform1f(point_size_location, 1.0f);
    }

    if (options.show_goldberg_grid) {
        glDepthFunc(GL_LEQUAL);
        glBindVertexArray(grid_ribbon_vao_);
        glDrawArrays(GL_TRIANGLES, 0, grid_ribbon_vertex_count_);

        glBindVertexArray(grid_ribbon_line_vao_);
        glDrawArrays(GL_LINES, 0, grid_ribbon_line_vertex_count_);
        glDepthFunc(GL_LESS);

        glUniform1f(point_size_location, 1.0f);
        glBindVertexArray(line_vao_);
        glDrawArrays(GL_LINES, 0, line_vertex_count_);
    }

    if (options.show_points) {
        glUniform1f(point_size_location, 6.0f);
        glBindVertexArray(point_vao_);
        glDrawArrays(GL_POINTS, 0, point_vertex_count_);
    }

    if (options.show_voxels) {
        const std::vector<DebugVertex> voxel_vertices = build_visible_voxel_vertices(current_svo_, view, aspect, far_plane);
        voxel_line_vertex_count_ = static_cast<int>(voxel_vertices.size());
        glBindVertexArray(voxel_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, voxel_vbo_);
        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(voxel_vertices.size() * sizeof(DebugVertex)),
            voxel_vertices.data(),
            GL_DYNAMIC_DRAW
        );
        glDrawArrays(GL_LINES, 0, voxel_line_vertex_count_);
    }

    const std::vector<DebugVertex> trail_vertices = build_spaceship_trail_vertices(ship);
    if (!trail_vertices.empty()) {
        glUniform1f(point_size_location, 1.0f);
        glBindVertexArray(overlay_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo_);
        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(trail_vertices.size() * sizeof(DebugVertex)),
            trail_vertices.data(),
            GL_DYNAMIC_DRAW
        );
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(trail_vertices.size()));
    }

    const std::vector<DebugVertex> ship_vertices = build_spaceship_vertices(ship);
    glBindVertexArray(overlay_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(ship_vertices.size() * sizeof(DebugVertex)),
        ship_vertices.data(),
        GL_DYNAMIC_DRAW
    );
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(ship_vertices.size()));

    glBindVertexArray(0);
    glUseProgram(0);

    if (show_fps) {
        render_fps_overlay(fps);
    }
}

void DebugRenderer::render_fps_overlay(float fps) {
    const std::vector<DebugVertex> vertices = build_fps_overlay_vertices(fps);

    glUseProgram(shader_);
    const Mat4 overlay_transform = identity();
    const int mvp_location = glGetUniformLocation(shader_, "u_mvp");
    const int point_size_location = glGetUniformLocation(shader_, "u_point_size");
    glUniformMatrix4fv(mvp_location, 1, GL_FALSE, overlay_transform.m);
    glUniform1f(point_size_location, 1.0f);

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(overlay_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(DebugVertex)), vertices.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices.size()));
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(0);
}

void DebugRenderer::release_mesh_buffers() {
    if (mesh_line_ebo_ != 0) {
        glDeleteBuffers(1, &mesh_line_ebo_);
        mesh_line_ebo_ = 0;
    }
    if (stitch_line_ebo_ != 0) {
        glDeleteBuffers(1, &stitch_line_ebo_);
        stitch_line_ebo_ = 0;
    }
    if (stitch_triangle_ebo_ != 0) {
        glDeleteBuffers(1, &stitch_triangle_ebo_);
        stitch_triangle_ebo_ = 0;
    }
    if (mesh_triangle_ebo_ != 0) {
        glDeleteBuffers(1, &mesh_triangle_ebo_);
        mesh_triangle_ebo_ = 0;
    }
    if (mesh_vbo_ != 0) {
        glDeleteBuffers(1, &mesh_vbo_);
        mesh_vbo_ = 0;
    }
    if (mesh_vao_ != 0) {
        glDeleteVertexArrays(1, &mesh_vao_);
        mesh_vao_ = 0;
    }
    if (voxel_vbo_ != 0) {
        glDeleteBuffers(1, &voxel_vbo_);
        voxel_vbo_ = 0;
    }
    if (voxel_vao_ != 0) {
        glDeleteVertexArrays(1, &voxel_vao_);
        voxel_vao_ = 0;
    }
    if (surface_net_ebo_ != 0) {
        glDeleteBuffers(1, &surface_net_ebo_);
        surface_net_ebo_ = 0;
    }
    if (surface_net_vbo_ != 0) {
        glDeleteBuffers(1, &surface_net_vbo_);
        surface_net_vbo_ = 0;
    }
    if (surface_net_vao_ != 0) {
        glDeleteVertexArrays(1, &surface_net_vao_);
        surface_net_vao_ = 0;
    }

    mesh_triangle_index_count_ = 0;
    mesh_line_index_count_ = 0;
    stitch_triangle_index_count_ = 0;
    stitch_line_index_count_ = 0;
    voxel_line_vertex_count_ = 0;
    surface_net_index_count_ = 0;
    current_svo_ = {};
}

void DebugRenderer::shutdown() {
    if (line_vbo_ != 0) {
        glDeleteBuffers(1, &line_vbo_);
        line_vbo_ = 0;
    }
    if (line_vao_ != 0) {
        glDeleteVertexArrays(1, &line_vao_);
        line_vao_ = 0;
    }
    if (point_vbo_ != 0) {
        glDeleteBuffers(1, &point_vbo_);
        point_vbo_ = 0;
    }
    if (point_vao_ != 0) {
        glDeleteVertexArrays(1, &point_vao_);
        point_vao_ = 0;
    }
    release_mesh_buffers();
    if (grid_ribbon_line_vbo_ != 0) {
        glDeleteBuffers(1, &grid_ribbon_line_vbo_);
        grid_ribbon_line_vbo_ = 0;
    }
    if (grid_ribbon_line_vao_ != 0) {
        glDeleteVertexArrays(1, &grid_ribbon_line_vao_);
        grid_ribbon_line_vao_ = 0;
    }
    if (grid_ribbon_vbo_ != 0) {
        glDeleteBuffers(1, &grid_ribbon_vbo_);
        grid_ribbon_vbo_ = 0;
    }
    if (grid_ribbon_vao_ != 0) {
        glDeleteVertexArrays(1, &grid_ribbon_vao_);
        grid_ribbon_vao_ = 0;
    }
    if (overlay_vbo_ != 0) {
        glDeleteBuffers(1, &overlay_vbo_);
        overlay_vbo_ = 0;
    }
    if (overlay_vao_ != 0) {
        glDeleteVertexArrays(1, &overlay_vao_);
        overlay_vao_ = 0;
    }
    if (shader_ != 0) {
        glDeleteProgram(shader_);
        shader_ = 0;
    }
    if (surface_net_shader_ != 0) {
        glDeleteProgram(surface_net_shader_);
        surface_net_shader_ = 0;
    }
    line_vertex_count_ = 0;
    point_vertex_count_ = 0;
    grid_ribbon_vertex_count_ = 0;
    grid_ribbon_line_vertex_count_ = 0;
}

} // namespace ae
