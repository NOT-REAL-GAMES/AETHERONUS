#include "aetheronus/debug_renderer.hpp"

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

uint32_t build_shader_program() {
    const uint32_t vertex = compile_shader(GL_VERTEX_SHADER, VertexShaderSource);
    const uint32_t fragment = compile_shader(GL_FRAGMENT_SHADER, FragmentShaderSource);
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

Vec3 point_color(const PointSample& point) {
    if (point.source_cell_id != point.owner_cell_id) {
        return {1.0f, 0.12f, 0.22f};
    }

    switch (point.material_id) {
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
        outer_loop.push_back(corner * RibbonRadius);
        inner_loop.push_back(normalize(lerp(corner, cell.center, InnerInset)) * RibbonRadius);
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

bool DebugRenderer::initialize(const GoldbergTopology& topology, const std::vector<PointSample>& points, const QuantizedMesh& mesh) {
    shutdown();

    shader_ = build_shader_program();
    if (shader_ == 0) {
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
    for (const PointSample& point : points) {
        point_vertices.push_back({point.position * 1.018f, point_color(point)});
    }

    std::vector<DebugVertex> mesh_vertices;
    mesh_vertices.reserve(mesh.vertices.size());
    for (const QuantizedMeshVertex& vertex : mesh.vertices) {
        const Vec3 color = vertex.material_id == 2u
            ? Vec3{0.045f, 0.095f, 0.045f}
            : vertex.material_id == 1u
            ? Vec3{0.20f, 0.15f, 0.10f}
            : Vec3{0.07f, 0.15f, 0.18f};
        mesh_vertices.push_back({vertex.position, color});
    }

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
    mesh_triangle_index_count_ = static_cast<int>(mesh.triangle_indices.size());
    mesh_line_index_count_ = static_cast<int>(mesh.line_indices.size());
    stitch_triangle_index_count_ = static_cast<int>(mesh.stitch_triangle_indices.size());
    stitch_line_index_count_ = static_cast<int>(mesh.stitch_line_indices.size());
    grid_ribbon_vertex_count_ = static_cast<int>(grid_ribbon_vertices.size());
    grid_ribbon_line_vertex_count_ = static_cast<int>(grid_ribbon_line_vertices.size());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.015f, 0.018f, 0.024f, 1.0f);
    return true;
}

void DebugRenderer::resize(int width, int height) {
    width_ = width > 0 ? width : 1;
    height_ = height > 0 ? height : 1;
}

void DebugRenderer::render(const OrbitCamera& camera, bool show_fps, float fps) {
    glViewport(0, 0, width_, height_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float cp = std::cos(camera.pitch);
    const Vec3 eye = {
        camera.distance * cp * std::sin(camera.yaw),
        camera.distance * std::sin(camera.pitch),
        camera.distance * cp * std::cos(camera.yaw),
    };

    const Mat4 projection = perspective(50.0f * Pi / 180.0f, static_cast<float>(width_) / static_cast<float>(height_), 0.05f, 32.0f);
    const Mat4 view = look_at(eye, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    const Mat4 mvp = projection * view;

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

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_line_ebo_);
    glDisableVertexAttribArray(1);
    glVertexAttrib3f(1, 0.42f, 0.92f, 1.0f);
    glDrawElements(GL_LINES, mesh_line_index_count_, GL_UNSIGNED_INT, nullptr);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, stitch_triangle_ebo_);
    glDrawElements(GL_TRIANGLES, stitch_triangle_index_count_, GL_UNSIGNED_INT, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, stitch_line_ebo_);
    glDisableVertexAttribArray(1);
    glVertexAttrib3f(1, 0.32f, 0.56f, 0.18f);
    glDrawElements(GL_LINES, stitch_line_index_count_, GL_UNSIGNED_INT, nullptr);
    glEnableVertexAttribArray(1);

    glDepthFunc(GL_LEQUAL);
    glBindVertexArray(grid_ribbon_vao_);
    glDrawArrays(GL_TRIANGLES, 0, grid_ribbon_vertex_count_);

    glBindVertexArray(grid_ribbon_line_vao_);
    glDrawArrays(GL_LINES, 0, grid_ribbon_line_vertex_count_);
    glDepthFunc(GL_LESS);

    glUniform1f(point_size_location, 1.0f);
    glBindVertexArray(line_vao_);
    glDrawArrays(GL_LINES, 0, line_vertex_count_);

    glUniform1f(point_size_location, 6.0f);
    glBindVertexArray(point_vao_);
    glDrawArrays(GL_POINTS, 0, point_vertex_count_);

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
    line_vertex_count_ = 0;
    point_vertex_count_ = 0;
    mesh_triangle_index_count_ = 0;
    mesh_line_index_count_ = 0;
    stitch_triangle_index_count_ = 0;
    stitch_line_index_count_ = 0;
    grid_ribbon_vertex_count_ = 0;
    grid_ribbon_line_vertex_count_ = 0;
}

} // namespace ae
