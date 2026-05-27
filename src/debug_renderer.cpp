#include "aetheronus/debug_renderer.hpp"

#include "aetheronus/planet_scale.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <array>
#include <chrono>
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

struct GpuSvoNode {
    uint32_t origin_size[4] = {};
    uint32_t meta[4] = {};
};

struct GpuCaveAnchor {
    float direction_radius[4] = {};
};

struct DrawArraysIndirectCommand {
    uint32_t count = 0;
    uint32_t prim_count = 1;
    uint32_t first = 0;
    uint32_t base_instance = 0;
};

static_assert(sizeof(GpuSvoNode) == 32);
static_assert(sizeof(GpuCaveAnchor) == 16);
static_assert(sizeof(DrawArraysIndirectCommand) == 16);

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

constexpr uint32_t OverlayDefaultVertexCapacity = 512u;
constexpr uint32_t VoxelComputeWorkgroupSize = 128u;
constexpr uint32_t VoxelComputeMaxBoxes = 131072u;
constexpr uint32_t CaveAnchorComputeWorkgroupSize = 128u;
constexpr uint32_t CaveAnchorComputeMaxVisible = 10000u;
constexpr uint32_t MaxTerrainHoles = 64u;
constexpr uint32_t MaxTerrainHeightMasks = 16u;

enum VoxelComputeFallbackCode : uint32_t {
    VoxelComputeFallbackNone = 0u,
    VoxelComputeFallbackUnavailable = 1u,
    VoxelComputeFallbackWaitingForSvo = 2u,
    VoxelComputeFallbackInvalidBuffers = 3u,
    VoxelComputeFallbackGlError = 4u,
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

uint32_t debug_hash_u32(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float debug_hash_unit(uint32_t value) {
    return static_cast<float>(debug_hash_u32(value) & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

const char* VertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_color;

uniform mat4 u_mvp;
uniform float u_point_size;
uniform int u_point_style;

out vec3 v_color;
out vec3 v_position;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    gl_PointSize = u_point_size;
    v_color = a_color;
    v_position = a_position;
}
)glsl";

const char* FragmentShaderSource = R"glsl(
#version 330 core
in vec3 v_color;
in vec3 v_position;
uniform int u_point_style;
uniform int u_terrain_hole_count;
uniform vec4 u_terrain_holes[64];
uniform int u_terrain_mask_count;
uniform usampler2DArray u_terrain_height_masks;
uniform vec4 u_terrain_mask_center_radius[16];
uniform vec4 u_terrain_mask_tangent[16];
uniform vec4 u_terrain_mask_bitangent[16];
out vec4 frag_color;

void main() {
    vec3 color = v_color;
    if (u_point_style == 0 && u_terrain_mask_count > 0) {
        vec3 surface_dir = normalize(v_position);
        for (int i = 0; i < u_terrain_mask_count; ++i) {
            vec3 mask_center = normalize(u_terrain_mask_center_radius[i].xyz);
            float mask_radius = u_terrain_mask_center_radius[i].w;
            vec3 rel = surface_dir - mask_center;
            float u = dot(rel, u_terrain_mask_tangent[i].xyz) / (mask_radius * 2.0) + 0.5;
            float v = dot(rel, u_terrain_mask_bitangent[i].xyz) / (mask_radius * 2.0) + 0.5;
            if (u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0) {
                ivec2 dims = textureSize(u_terrain_height_masks, 0).xy;
                ivec2 texel = clamp(ivec2(vec2(u, v) * vec2(dims - ivec2(1))), ivec2(0), dims - ivec2(1));
                uint h = texelFetch(u_terrain_height_masks, ivec3(texel, i), 0).r;
                if (h == 0u) {
                    discard;
                }
                uint h0 = texelFetch(u_terrain_height_masks, ivec3(clamp(texel + ivec2(-1, 0), ivec2(0), dims - ivec2(1)), i), 0).r;
                uint h1 = texelFetch(u_terrain_height_masks, ivec3(clamp(texel + ivec2(1, 0), ivec2(0), dims - ivec2(1)), i), 0).r;
                uint h2 = texelFetch(u_terrain_height_masks, ivec3(clamp(texel + ivec2(0, -1), ivec2(0), dims - ivec2(1)), i), 0).r;
                uint h3 = texelFetch(u_terrain_height_masks, ivec3(clamp(texel + ivec2(0, 1), ivec2(0), dims - ivec2(1)), i), 0).r;
                if (h0 == 0u || h1 == 0u || h2 == 0u || h3 == 0u) {
                    color = mix(vec3(0.020, 0.012, 0.007), color, 0.52);
                }
            }
        }
    }
    if (u_point_style == 0 && u_terrain_hole_count > 0) {
        vec3 surface_dir = normalize(v_position);
        float best_edge = 100000.0;
        for (int i = 0; i < u_terrain_hole_count; ++i) {
            vec3 hole_dir = normalize(u_terrain_holes[i].xyz);
            float chord_radius = u_terrain_holes[i].w;
            vec3 delta = surface_dir - hole_dir;
            float dist = length(delta);
            if (dot(surface_dir, hole_dir) > 0.0) {
                best_edge = min(best_edge, dist / max(chord_radius, 0.000001));
            }
        }
        if (best_edge <= 0.94) {
            discard;
        }
        if (best_edge <= 1.08) {
            float blend = smoothstep(0.94, 1.08, best_edge);
            vec3 cave_shadow = vec3(0.020, 0.012, 0.007);
            vec3 weathered_rim = vec3(0.16, 0.095, 0.045);
            color = mix(mix(cave_shadow, weathered_rim, blend), color, blend * 0.86);
        }
    }

    if (u_point_style == 1) {
        vec2 p = gl_PointCoord * 2.0 - vec2(1.0);
        float r2 = dot(p, p);
        if (r2 > 1.0) {
            discard;
        }
        float r = sqrt(r2);
        float rim = smoothstep(0.62, 0.98, r);
        float center_shadow = 1.0 - smoothstep(0.0, 0.72, r);
        vec3 cave_center = vec3(0.010, 0.007, 0.004);
        vec3 cave_rim = color;
        vec3 color = mix(cave_center, cave_rim, rim * 0.72);
        color *= 0.72 + center_shadow * 0.28;
        frag_color = vec4(color, 1.0);
        return;
    }
    frag_color = vec4(color, 1.0);
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

const char* VoxelDebugComputeShaderSource = R"glsl(
#version 430 core
layout(local_size_x = 128) in;

struct SvoNode {
    uvec4 origin_size;
    uvec4 meta;
};

layout(std430, binding = 0) readonly buffer SvoNodes {
    SvoNode nodes[];
};

layout(std430, binding = 1) writeonly buffer VoxelVertices {
    float vertex_data[];
};

layout(std430, binding = 2) buffer VoxelCounters {
    uint emitted_boxes;
};

layout(std430, binding = 3) buffer VoxelDrawCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint base_instance;
};

uniform uint u_node_count;
uniform float u_bounds_radius;
uniform uint u_svo_depth;
uniform uint u_max_boxes;
uniform float u_planet_radius;
uniform vec3 u_eye;
uniform vec3 u_range_center;
uniform vec3 u_forward;
uniform vec3 u_right;
uniform vec3 u_up;
uniform float u_tan_half_fov_y;
uniform float u_tan_half_fov_x;
uniform float u_near_plane;
uniform float u_far_plane;
uniform float u_occlusion_radius;
uniform float u_max_distance;
uniform float u_range_radius;
uniform float u_debug_box_size;

const vec3 VoxelColor = vec3(0.95, 0.55, 0.14);

vec3 node_center_mesh(SvoNode node) {
    float leaf_count = float(1u << u_svo_depth);
    float cell_size = (u_bounds_radius * 2.0) / leaf_count;
    vec3 origin = vec3(node.origin_size.xyz);
    float size = float(node.origin_size.w);
    return vec3(-u_bounds_radius) + (origin + vec3(size * 0.5)) * cell_size;
}

float node_world_size(SvoNode node) {
    float leaf_count = float(1u << u_svo_depth);
    float cell_size = (u_bounds_radius * 2.0) / leaf_count;
    return float(node.origin_size.w) * cell_size * u_planet_radius;
}

bool voxel_box_visible(vec3 center, float radius) {
    if (length(center - u_range_center) > u_range_radius + radius) {
        return false;
    }

    vec3 relative = center - u_eye;
    float z = dot(relative, u_forward);
    if (z + radius < u_near_plane || z - radius > u_far_plane) {
        return false;
    }

    float x = dot(relative, u_right);
    float y = dot(relative, u_up);
    float perspective_z = max(z, u_near_plane);
    if (abs(x) - radius > perspective_z * u_tan_half_fov_x ||
        abs(y) - radius > perspective_z * u_tan_half_fov_y) {
        return false;
    }

    float eye_radius = length(u_eye);
    if (eye_radius > u_occlusion_radius + 1.0) {
        float horizon_limit = u_occlusion_radius * u_occlusion_radius;
        float box_margin = radius * eye_radius;
        if (dot(center, u_eye) + box_margin < horizon_limit) {
            return false;
        }
    }

    return true;
}

void write_vertex(uint index, vec3 position, vec3 color) {
    uint base = index * 6u;
    vertex_data[base + 0u] = position.x;
    vertex_data[base + 1u] = position.y;
    vertex_data[base + 2u] = position.z;
    vertex_data[base + 3u] = color.x;
    vertex_data[base + 4u] = color.y;
    vertex_data[base + 5u] = color.z;
}

void write_line(inout uint cursor, vec3 a, vec3 b) {
    write_vertex(cursor++, a, VoxelColor);
    write_vertex(cursor++, b, VoxelColor);
}

void emit_wire_cube(vec3 center, float size) {
    uint box_index = atomicAdd(emitted_boxes, 1u);
    if (box_index >= u_max_boxes) {
        return;
    }

    atomicAdd(vertex_count, 24u);
    uint cursor = box_index * 24u;
    vec3 h = vec3(size * 0.5);
    vec3 p000 = center + vec3(-h.x, -h.y, -h.z);
    vec3 p100 = center + vec3( h.x, -h.y, -h.z);
    vec3 p010 = center + vec3(-h.x,  h.y, -h.z);
    vec3 p110 = center + vec3( h.x,  h.y, -h.z);
    vec3 p001 = center + vec3(-h.x, -h.y,  h.z);
    vec3 p101 = center + vec3( h.x, -h.y,  h.z);
    vec3 p011 = center + vec3(-h.x,  h.y,  h.z);
    vec3 p111 = center + vec3( h.x,  h.y,  h.z);

    write_line(cursor, p000, p100);
    write_line(cursor, p100, p110);
    write_line(cursor, p110, p010);
    write_line(cursor, p010, p000);
    write_line(cursor, p001, p101);
    write_line(cursor, p101, p111);
    write_line(cursor, p111, p011);
    write_line(cursor, p011, p001);
    write_line(cursor, p000, p001);
    write_line(cursor, p100, p101);
    write_line(cursor, p110, p111);
    write_line(cursor, p010, p011);
}

int first_step(float minimum, float range_minimum, float box_size, uint steps) {
    return clamp(int(floor((range_minimum - minimum) / box_size)), 0, int(steps) - 1);
}

int last_step(float minimum, float range_maximum, float box_size, uint steps) {
    return clamp(int(ceil((range_maximum - minimum) / box_size)), 0, int(steps) - 1);
}

void emit_subdivided_wire_cubes(vec3 center, float size) {
    float target_size = max(0.25, u_debug_box_size);
    uint steps = max(1u, uint(ceil(size / target_size)));
    float box_size = size / float(steps);
    float box_radius = box_size * 0.8661;
    vec3 min_corner = center - vec3(size * 0.5);
    vec3 range_min = u_range_center - vec3(u_range_radius + box_radius);
    vec3 range_max = u_range_center + vec3(u_range_radius + box_radius);

    int ix0 = first_step(min_corner.x, range_min.x, box_size, steps);
    int iy0 = first_step(min_corner.y, range_min.y, box_size, steps);
    int iz0 = first_step(min_corner.z, range_min.z, box_size, steps);
    int ix1 = last_step(min_corner.x, range_max.x, box_size, steps);
    int iy1 = last_step(min_corner.y, range_max.y, box_size, steps);
    int iz1 = last_step(min_corner.z, range_max.z, box_size, steps);

    for (int z = iz0; z <= iz1; ++z) {
        for (int y = iy0; y <= iy1; ++y) {
            for (int x = ix0; x <= ix1; ++x) {
                if (emitted_boxes >= u_max_boxes) {
                    return;
                }
                vec3 box_center = min_corner + (vec3(float(x), float(y), float(z)) + vec3(0.5)) * box_size;
                if (voxel_box_visible(box_center, box_radius)) {
                    emit_wire_cube(box_center, box_size);
                }
            }
        }
    }
}

void main() {
    uint node_index = gl_GlobalInvocationID.x;
    if (node_index >= u_node_count || u_max_boxes == 0u) {
        return;
    }

    SvoNode node = nodes[node_index];
    vec3 center = node_center_mesh(node) * u_planet_radius;
    float size = node_world_size(node);
    float radius = size * 0.8661;
    if (!voxel_box_visible(center, radius)) {
        return;
    }

    if (size > u_debug_box_size * 1.5) {
        emit_subdivided_wire_cubes(center, size);
    } else {
        emit_wire_cube(center, size);
    }
}
)glsl";

const char* CaveAnchorComputeShaderSource = R"glsl(
#version 430 core
layout(local_size_x = 128) in;

struct CaveAnchor {
    vec4 direction_radius;
};

layout(std430, binding = 4) readonly buffer CaveAnchors {
    CaveAnchor anchors[];
};

layout(std430, binding = 5) writeonly buffer CaveVertices {
    float vertex_data[];
};

layout(std430, binding = 6) buffer CaveCounters {
    uint emitted_points;
};

layout(std430, binding = 7) buffer CaveDrawCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint base_instance;
};

uniform uint u_anchor_count;
uniform uint u_max_visible;
uniform float u_planet_radius;
uniform vec3 u_eye;
uniform vec3 u_forward;
uniform vec3 u_right;
uniform vec3 u_up;
uniform float u_tan_half_fov_y;
uniform float u_tan_half_fov_x;
uniform float u_near_plane;
uniform float u_far_plane;
uniform float u_occlusion_radius;

const vec3 CaveColor = vec3(0.23, 0.13, 0.055);

bool cave_visible(vec3 center, float radius) {
    vec3 relative = center - u_eye;
    if (length(relative) > u_max_distance + radius) {
        return false;
    }

    float z = dot(relative, u_forward);
    if (z + radius < u_near_plane || z - radius > u_far_plane) {
        return false;
    }

    float x = dot(relative, u_right);
    float y = dot(relative, u_up);
    float perspective_z = max(z, u_near_plane);
    if (abs(x) - radius > perspective_z * u_tan_half_fov_x ||
        abs(y) - radius > perspective_z * u_tan_half_fov_y) {
        return false;
    }

    float eye_radius = length(u_eye);
    if (eye_radius > u_occlusion_radius + 1.0) {
        float horizon_limit = u_occlusion_radius * u_occlusion_radius;
        float point_margin = radius * eye_radius;
        if (dot(center, u_eye) + point_margin < horizon_limit) {
            return false;
        }
    }
    return true;
}

void write_vertex(uint index, vec3 position, vec3 color) {
    uint base = index * 6u;
    vertex_data[base + 0u] = position.x;
    vertex_data[base + 1u] = position.y;
    vertex_data[base + 2u] = position.z;
    vertex_data[base + 3u] = color.x;
    vertex_data[base + 4u] = color.y;
    vertex_data[base + 5u] = color.z;
}

void main() {
    uint anchor_index = gl_GlobalInvocationID.x;
    if (anchor_index >= u_anchor_count || u_max_visible == 0u) {
        return;
    }

    vec3 direction = normalize(anchors[anchor_index].direction_radius.xyz);
    vec3 center = direction * (u_planet_radius * 1.004);
    float radius = max(18.0, anchors[anchor_index].direction_radius.w);
    if (!cave_visible(center, radius)) {
        return;
    }

    uint point_index = atomicAdd(emitted_points, 1u);
    if (point_index >= u_max_visible) {
        return;
    }
    atomicAdd(vertex_count, 1u);
    write_vertex(point_index, center, CaveColor);
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

uint32_t build_compute_shader_program(const char* compute_source) {
    const uint32_t compute = compile_shader(GL_COMPUTE_SHADER, compute_source);
    if (compute == 0) {
        return 0;
    }

    const uint32_t program = glCreateProgram();
    glAttachShader(program, compute);
    glLinkProgram(program);
    glDeleteShader(compute);

    int ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = {};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Compute shader link failed: " << log << std::endl;
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

uint32_t build_voxel_compute_shader_program() {
    return build_compute_shader_program(VoxelDebugComputeShaderSource);
}

uint32_t build_cave_anchor_compute_shader_program() {
    return build_compute_shader_program(CaveAnchorComputeShaderSource);
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

VoxelDebugView make_voxel_debug_view(const CameraView& camera, float aspect, float far_plane) {
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
    return view;
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

    const VoxelDebugView view = make_voxel_debug_view(camera, aspect, far_plane);

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
            : vertex.material_id == 6u
            ? Vec3{0.30f, 0.18f, 0.08f}
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

std::vector<DebugVertex> build_cave_anchor_vertices(const QuantizedMesh& mesh) {
    std::vector<DebugVertex> vertices;
    vertices.reserve(mesh.cave_anchor_points.size());
    constexpr float AnchorSurfaceLift = 1.004f;
    constexpr Vec3 AnchorColor = {0.23f, 0.13f, 0.055f};
    for (Vec3 anchor : mesh.cave_anchor_points) {
        vertices.push_back({planet_to_world(normalize(anchor) * AnchorSurfaceLift), AnchorColor});
    }
    return vertices;
}

std::vector<DebugVertex> build_cave_transition_vertices(const std::vector<LocalVoxelFeature>& holes) {
    constexpr uint32_t Segments = 28u;
    constexpr Vec3 OuterColor = {0.070f, 0.150f, 0.175f};
    constexpr Vec3 MidColor = {0.19f, 0.095f, 0.034f};
    constexpr Vec3 InnerColor = {0.018f, 0.010f, 0.004f};
    std::vector<DebugVertex> vertices;
    vertices.reserve(holes.size() * Segments * 12u);

    auto append_triangle = [&vertices](DebugVertex a, DebugVertex b, DebugVertex c) {
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
    };

    for (const LocalVoxelFeature& hole : holes) {
        if (hole.kind != VoxelFeatureKind::CaveSystem || hole.entrance_radius_km <= 0.0f || length(hole.center_mesh) <= 0.000001f) {
            continue;
        }

        const Vec3 normal = normalize(hole.center_mesh);
        const Vec3 fallback_axis = std::fabs(normal.y) < 0.92f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
        const Vec3 tangent = length(hole.tangent_mesh) > 0.000001f ? normalize(hole.tangent_mesh) : normalize(cross(fallback_axis, normal));
        const Vec3 bitangent = length(hole.bitangent_mesh) > 0.000001f ? normalize(hole.bitangent_mesh) : normalize(cross(normal, tangent));
        const float radius = kilometers_to_world_units(hole.entrance_radius_km);
        const std::array<float, 3> radii = {radius * 0.91f, radius * 1.005f, radius * 1.12f};
        const std::array<float, 3> lifts = {1.00008f, 1.00018f, 1.00030f};
        const std::array<Vec3, 3> colors = {InnerColor, MidColor, OuterColor};
        const uint32_t seed = hole.seed ^ (hole.feature_id * 0x9e3779b9u) ^ 0x4cf5ad43u;

        auto ring_vertex = [&](uint32_t ring, uint32_t segment) {
            const float t = static_cast<float>(segment) / static_cast<float>(Segments);
            const float angle = t * Pi * 2.0f;
            const float wave = std::sin(angle * 3.0f + debug_hash_unit(seed ^ 0x51ed270bu) * Pi * 2.0f) * 0.020f;
            const float chip = debug_hash_unit(seed ^ (segment * 0x85ebca6bu) ^ (ring * 0x27d4eb2du)) - 0.5f;
            const float radial_jitter = 1.0f + wave + chip * (ring == 2u ? 0.075f : 0.045f);
            const Vec3 local =
                tangent * (std::cos(angle) * radii[ring] * radial_jitter) +
                bitangent * (std::sin(angle) * radii[ring] * radial_jitter);
            const Vec3 position = normalize(normal + local) * lifts[ring];
            return DebugVertex{planet_to_world(position), colors[ring]};
        };

        for (uint32_t segment = 0; segment < Segments; ++segment) {
            const uint32_t next = segment + 1u;
            const DebugVertex inner_a = ring_vertex(0u, segment);
            const DebugVertex inner_b = ring_vertex(0u, next);
            const DebugVertex mid_a = ring_vertex(1u, segment);
            const DebugVertex mid_b = ring_vertex(1u, next);
            const DebugVertex outer_a = ring_vertex(2u, segment);
            const DebugVertex outer_b = ring_vertex(2u, next);

            append_triangle(inner_a, mid_a, inner_b);
            append_triangle(inner_b, mid_a, mid_b);
            append_triangle(mid_a, outer_a, mid_b);
            append_triangle(mid_b, outer_a, outer_b);
        }
    }

    return vertices;
}

std::vector<DebugVertex> build_terrain_mask_transition_vertices(const std::vector<TerrainHeightMask>& masks) {
    constexpr Vec3 EdgeColor = {0.020f, 0.012f, 0.007f};
    constexpr Vec3 RimColor = {0.13f, 0.070f, 0.030f};
    std::vector<DebugVertex> vertices;

    auto append_quad = [&vertices](DebugVertex a, DebugVertex b, DebugVertex c, DebugVertex d) {
        vertices.push_back(a);
        vertices.push_back(b);
        vertices.push_back(c);
        vertices.push_back(a);
        vertices.push_back(c);
        vertices.push_back(d);
    };

    auto mask_point = [](const TerrainHeightMask& mask, float px, float py, float lift, Vec3 color) {
        const float resolution = static_cast<float>(std::max(2u, mask.resolution));
        const float radius_mesh = kilometers_to_world_units(mask.radius_km);
        const float local_x = ((px / (resolution - 1.0f)) - 0.5f) * radius_mesh * 2.0f;
        const float local_y = ((py / (resolution - 1.0f)) - 0.5f) * radius_mesh * 2.0f;
        const Vec3 position = normalize(normalize(mask.center_mesh) + mask.tangent_mesh * local_x + mask.bitangent_mesh * local_y) * lift;
        return DebugVertex{planet_to_world(position), color};
    };

    for (const TerrainHeightMask& mask : masks) {
        if (mask.resolution < 2u ||
            mask.heights.size() != static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution) ||
            mask.radius_km <= 0.0f) {
            continue;
        }
        constexpr float OuterWidth = 0.58f;
        constexpr float InnerWidth = 0.34f;
        vertices.reserve(vertices.size() + mask.resolution * 48u);
        auto solid = [&](int32_t x, int32_t y) {
            if (x < 0 || y < 0 || x >= static_cast<int32_t>(mask.resolution) || y >= static_cast<int32_t>(mask.resolution)) {
                return true;
            }
            return mask.heights[static_cast<size_t>(y) * mask.resolution + x] != 0u;
        };

        auto append_edge_strip = [&](float x0, float y0, float x1, float y1, float normal_x, float normal_y) {
            const DebugVertex outer_a = mask_point(
                mask,
                x0 + normal_x * OuterWidth,
                y0 + normal_y * OuterWidth,
                1.00028f,
                RimColor
            );
            const DebugVertex outer_b = mask_point(
                mask,
                x1 + normal_x * OuterWidth,
                y1 + normal_y * OuterWidth,
                1.00028f,
                RimColor
            );
            const DebugVertex inner_b = mask_point(
                mask,
                x1 - normal_x * InnerWidth,
                y1 - normal_y * InnerWidth,
                1.00008f,
                EdgeColor
            );
            const DebugVertex inner_a = mask_point(
                mask,
                x0 - normal_x * InnerWidth,
                y0 - normal_y * InnerWidth,
                1.00008f,
                EdgeColor
            );
            append_quad(outer_a, outer_b, inner_b, inner_a);
        };

        for (uint32_t y = 0u; y < mask.resolution; ++y) {
            for (uint32_t x = 0u; x < mask.resolution; ++x) {
                if (solid(static_cast<int32_t>(x), static_cast<int32_t>(y))) {
                    continue;
                }
                const int32_t xi = static_cast<int32_t>(x);
                const int32_t yi = static_cast<int32_t>(y);
                const float left = static_cast<float>(x) - 0.5f;
                const float right = static_cast<float>(x) + 0.5f;
                const float bottom = static_cast<float>(y) - 0.5f;
                const float top = static_cast<float>(y) + 0.5f;
                if (solid(xi - 1, yi)) {
                    append_edge_strip(left, bottom, left, top, -1.0f, 0.0f);
                }
                if (solid(xi + 1, yi)) {
                    append_edge_strip(right, top, right, bottom, 1.0f, 0.0f);
                }
                if (solid(xi, yi - 1)) {
                    append_edge_strip(right, bottom, left, bottom, 0.0f, -1.0f);
                }
                if (solid(xi, yi + 1)) {
                    append_edge_strip(left, top, right, top, 0.0f, 1.0f);
                }
            }
        }
    }
    return vertices;
}

std::vector<SurfaceNetVertex> build_surface_net_vertices(const SurfaceNetMesh& surface_net) {
    std::vector<SurfaceNetVertex> vertices;
    vertices.reserve(surface_net.vertices.size());
    auto depth_color = [](uint32_t depth) {
        if (depth >= 16u) {
            return Vec3{0.18f, 0.92f, 1.00f};
        }
        if (depth >= 15u) {
            return Vec3{0.32f, 0.70f, 1.00f};
        }
        if (depth >= 13u) {
            return Vec3{0.75f, 0.42f, 1.00f};
        }
        if (depth >= 9u) {
            return Vec3{1.00f, 0.82f, 0.24f};
        }
        return Vec3{0.95f, 0.50f, 0.12f};
    };
    for (uint32_t i = 0; i < surface_net.vertices.size(); ++i) {
        const Vec3 normal = i < surface_net.normals.size() ? normalize(surface_net.normals[i]) : normalize(surface_net.vertices[i]);
        const uint32_t depth = i < surface_net.vertex_depths.size() ? surface_net.vertex_depths[i] : surface_net.source_depth;
        const Vec3 color = surface_net.material_id == 5u ? depth_color(depth) : Vec3{0.86f, 0.74f, 0.42f};
        vertices.push_back({planet_to_world(surface_net.vertices[i]), normal, color});
    }
    return vertices;
}

Vec3 ship_forward(const SpaceshipState& ship) {
    return normalize(ship.forward);
}

void append_ship_triangle(DebugVertex* vertices, uint32_t& count, Vec3 a, Vec3 b, Vec3 c, Vec3 color) {
    vertices[count++] = {a, color};
    vertices[count++] = {b, color};
    vertices[count++] = {c, color};
}

uint32_t build_spaceship_vertices(const SpaceshipState& ship, DebugVertex* vertices) {
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

    uint32_t count = 0;
    append_ship_triangle(vertices, count, nose, left, top, {0.82f, 0.92f, 1.0f});
    append_ship_triangle(vertices, count, right_wing, nose, top, {0.46f, 0.76f, 1.0f});
    append_ship_triangle(vertices, count, left, tail, top, {0.28f, 0.42f, 0.72f});
    append_ship_triangle(vertices, count, tail, right_wing, top, {0.22f, 0.36f, 0.66f});
    append_ship_triangle(vertices, count, nose, keel, left, {0.12f, 0.18f, 0.28f});
    append_ship_triangle(vertices, count, right_wing, keel, nose, {0.10f, 0.16f, 0.26f});
    return count;
}

uint32_t build_spaceship_trail_vertices(const SpaceshipState& ship, DebugVertex* vertices) {
    if (ship.trail_count < 2) {
        return 0;
    }

    uint32_t count = 0;
    const uint32_t first_index = (ship.trail_head + SpaceshipState::TrailCapacity - ship.trail_count) % SpaceshipState::TrailCapacity;
    auto trail_point = [&](uint32_t index) {
        return ship.trail[(first_index + index) % SpaceshipState::TrailCapacity];
    };
    for (uint32_t i = 1; i < ship.trail_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(ship.trail_count - 1u);
        const Vec3 color = {0.16f + t * 0.42f, 0.38f + t * 0.42f, 0.58f + t * 0.38f};
        vertices[count++] = {trail_point(i - 1u), color};
        vertices[count++] = {trail_point(i), color};
    }
    return count;
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

std::vector<DebugVertex> build_progress_overlay_vertices(double progress) {
    std::vector<DebugVertex> vertices;
    const Vec3 border_color = {0.20f, 0.34f, 0.42f};
    const Vec3 fill_color = {0.18f, 0.92f, 1.0f};
    const Vec3 digit_color = {0.74f, 0.96f, 1.0f};
    const float left = -0.62f;
    const float right = 0.46f;
    const float top = -0.78f;
    const float bottom = -0.86f;
    const double clamped = std::clamp(progress, 0.0, 1.0);

    append_line(vertices, left, top, right, top, border_color);
    append_line(vertices, right, top, right, bottom, border_color);
    append_line(vertices, right, bottom, left, bottom, border_color);
    append_line(vertices, left, bottom, left, top, border_color);

    constexpr uint32_t FillSegments = 96;
    const uint32_t lit_segments = static_cast<uint32_t>(std::round(clamped * static_cast<double>(FillSegments)));
    for (uint32_t i = 0; i < lit_segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(FillSegments);
        const float x = left + (right - left) * t;
        append_line(vertices, x, top - 0.012f, x, bottom + 0.012f, fill_color);
    }

    char buffer[16] = {};
    std::snprintf(buffer, sizeof(buffer), "%010.6f", clamped * 100.0);
    float digit_x = 0.39f;
    for (const char* cursor = buffer; *cursor != '\0'; ++cursor) {
        if (*cursor == '.') {
            append_line(vertices, digit_x + 0.008f, bottom + 0.006f, digit_x + 0.012f, bottom + 0.006f, digit_color);
            append_line(vertices, digit_x + 0.010f, bottom + 0.010f, digit_x + 0.010f, bottom + 0.002f, digit_color);
            digit_x += 0.018f;
            continue;
        }
        append_seven_segment(vertices, *cursor - '0', digit_x, top + 0.01f, 0.025f, digit_color);
        digit_x += 0.038f;
    }

    return vertices;
}

void upload_dynamic_overlay_vertices(uint32_t& capacity_bytes, const DebugVertex* vertices, size_t vertex_count) {
    const uint32_t byte_count = static_cast<uint32_t>(vertex_count * sizeof(DebugVertex));
    if (byte_count <= capacity_bytes) {
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(byte_count), vertices);
        return;
    }
    capacity_bytes = std::max(byte_count, OverlayDefaultVertexCapacity * static_cast<uint32_t>(sizeof(DebugVertex)));
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(capacity_bytes), nullptr, GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(byte_count), vertices);
}

} // namespace

DebugRenderer::~DebugRenderer() {
    shutdown();
}

bool DebugRenderer::initialize_voxel_compute() {
    voxel_compute_available_ = false;
    voxel_compute_enabled_ = false;
    if (!GLAD_GL_VERSION_4_3) {
        return false;
    }

    voxel_compute_shader_ = build_voxel_compute_shader_program();
    if (voxel_compute_shader_ == 0) {
        std::cerr << "Voxel compute shader unavailable; using CPU voxel debug fallback." << std::endl;
        return false;
    }

    voxel_compute_node_count_location_ = glGetUniformLocation(voxel_compute_shader_, "u_node_count");
    voxel_compute_bounds_radius_location_ = glGetUniformLocation(voxel_compute_shader_, "u_bounds_radius");
    voxel_compute_svo_depth_location_ = glGetUniformLocation(voxel_compute_shader_, "u_svo_depth");
    voxel_compute_max_boxes_location_ = glGetUniformLocation(voxel_compute_shader_, "u_max_boxes");
    voxel_compute_planet_radius_location_ = glGetUniformLocation(voxel_compute_shader_, "u_planet_radius");
    voxel_compute_eye_location_ = glGetUniformLocation(voxel_compute_shader_, "u_eye");
    voxel_compute_range_center_location_ = glGetUniformLocation(voxel_compute_shader_, "u_range_center");
    voxel_compute_forward_location_ = glGetUniformLocation(voxel_compute_shader_, "u_forward");
    voxel_compute_right_location_ = glGetUniformLocation(voxel_compute_shader_, "u_right");
    voxel_compute_up_location_ = glGetUniformLocation(voxel_compute_shader_, "u_up");
    voxel_compute_tan_half_fov_y_location_ = glGetUniformLocation(voxel_compute_shader_, "u_tan_half_fov_y");
    voxel_compute_tan_half_fov_x_location_ = glGetUniformLocation(voxel_compute_shader_, "u_tan_half_fov_x");
    voxel_compute_near_plane_location_ = glGetUniformLocation(voxel_compute_shader_, "u_near_plane");
    voxel_compute_far_plane_location_ = glGetUniformLocation(voxel_compute_shader_, "u_far_plane");
    voxel_compute_occlusion_radius_location_ = glGetUniformLocation(voxel_compute_shader_, "u_occlusion_radius");
    voxel_compute_range_radius_location_ = glGetUniformLocation(voxel_compute_shader_, "u_range_radius");
    voxel_compute_debug_box_size_location_ = glGetUniformLocation(voxel_compute_shader_, "u_debug_box_size");

    glGenBuffers(1, &voxel_node_ssbo_);
    glGenBuffers(1, &voxel_counter_ssbo_);
    glGenBuffers(1, &voxel_indirect_buffer_);
    if (voxel_node_ssbo_ == 0 || voxel_counter_ssbo_ == 0 || voxel_indirect_buffer_ == 0) {
        std::cerr << "Voxel compute buffers unavailable; using CPU voxel debug fallback." << std::endl;
        return false;
    }

    uint32_t zero_counter = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, voxel_counter_ssbo_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), &zero_counter, GL_DYNAMIC_DRAW);
    DrawArraysIndirectCommand command;
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, voxel_indirect_buffer_);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawArraysIndirectCommand), &command, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "Voxel compute initialization GL error 0x" << std::hex << error << std::dec
                  << "; using CPU voxel debug fallback." << std::endl;
        return false;
    }

    voxel_compute_available_ = true;
    voxel_compute_enabled_ = true;
    return true;
}

bool DebugRenderer::initialize_cave_anchor_compute() {
    cave_anchor_compute_available_ = false;
    cave_anchor_compute_enabled_ = false;
    if (!GLAD_GL_VERSION_4_3) {
        return false;
    }

    cave_anchor_compute_shader_ = build_cave_anchor_compute_shader_program();
    if (cave_anchor_compute_shader_ == 0) {
        std::cerr << "Cave anchor compute shader unavailable; using CPU cave mouth fallback." << std::endl;
        return false;
    }

    cave_anchor_compute_anchor_count_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_anchor_count");
    cave_anchor_compute_max_visible_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_max_visible");
    cave_anchor_compute_planet_radius_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_planet_radius");
    cave_anchor_compute_eye_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_eye");
    cave_anchor_compute_forward_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_forward");
    cave_anchor_compute_right_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_right");
    cave_anchor_compute_up_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_up");
    cave_anchor_compute_tan_half_fov_y_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_tan_half_fov_y");
    cave_anchor_compute_tan_half_fov_x_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_tan_half_fov_x");
    cave_anchor_compute_near_plane_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_near_plane");
    cave_anchor_compute_far_plane_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_far_plane");
    cave_anchor_compute_occlusion_radius_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_occlusion_radius");
    cave_anchor_compute_max_distance_location_ = glGetUniformLocation(cave_anchor_compute_shader_, "u_max_distance");

    glGenBuffers(1, &cave_anchor_ssbo_);
    glGenBuffers(1, &cave_anchor_counter_ssbo_);
    glGenBuffers(1, &cave_anchor_indirect_buffer_);
    if (cave_anchor_ssbo_ == 0 || cave_anchor_counter_ssbo_ == 0 || cave_anchor_indirect_buffer_ == 0) {
        std::cerr << "Cave anchor compute buffers unavailable; using CPU cave mouth fallback." << std::endl;
        return false;
    }

    uint32_t zero_counter = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cave_anchor_counter_ssbo_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), &zero_counter, GL_DYNAMIC_DRAW);
    DrawArraysIndirectCommand command;
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cave_anchor_indirect_buffer_);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawArraysIndirectCommand), &command, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "Cave anchor compute initialization GL error 0x" << std::hex << error << std::dec
                  << "; using CPU cave mouth fallback." << std::endl;
        return false;
    }

    cave_anchor_compute_available_ = true;
    cave_anchor_compute_enabled_ = true;
    return true;
}

void DebugRenderer::upload_voxel_compute_buffers(const SparseVoxelOctree& svo) {
    voxel_compute_node_count_ = 0;
    voxel_compute_max_boxes_ = 0;
    voxel_compute_vertex_capacity_ = 0;
    if (!voxel_compute_available_ || voxel_node_ssbo_ == 0 || voxel_vbo_ == 0 || svo.nodes.empty() || svo.bounds_radius <= 0.0f || svo.depth == 0u) {
        perf_stats_.voxel_compute_fallback_code = voxel_compute_available_
            ? VoxelComputeFallbackWaitingForSvo
            : VoxelComputeFallbackUnavailable;
        return;
    }

    std::vector<GpuSvoNode> gpu_nodes;
    gpu_nodes.reserve(std::min<size_t>(svo.nodes.size(), std::max(1u, svo.debug_box_count)));
    for (const SparseVoxelOctreeNode& node : svo.nodes) {
        if (node.depth > svo.debug_draw_depth || (node.child_mask != 0u && node.depth < svo.debug_draw_depth)) {
            continue;
        }
        GpuSvoNode gpu_node;
        gpu_node.origin_size[0] = node.origin_x;
        gpu_node.origin_size[1] = node.origin_y;
        gpu_node.origin_size[2] = node.origin_z;
        gpu_node.origin_size[3] = node.size;
        gpu_node.meta[0] = node.child_mask;
        gpu_node.meta[1] = node.first_child;
        gpu_node.meta[2] = node.depth;
        gpu_node.meta[3] = node.occupied_leaf_count;
        gpu_nodes.push_back(gpu_node);
    }
    if (gpu_nodes.empty()) {
        perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackWaitingForSvo;
        return;
    }

    voxel_compute_node_count_ = static_cast<uint32_t>(gpu_nodes.size());
    voxel_compute_max_boxes_ = std::max(1u, std::min(svo.debug_max_boxes, VoxelComputeMaxBoxes));
    voxel_compute_vertex_capacity_ = voxel_compute_max_boxes_ * 24u;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, voxel_node_ssbo_);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(gpu_nodes.size() * sizeof(GpuSvoNode)),
        gpu_nodes.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ARRAY_BUFFER, voxel_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(static_cast<uint64_t>(voxel_compute_vertex_capacity_) * sizeof(DebugVertex)),
        nullptr,
        GL_DYNAMIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackNone;
}

void DebugRenderer::upload_cave_anchor_compute_buffers(const QuantizedMesh& mesh) {
    cave_anchor_compute_count_ = 0;
    cave_anchor_compute_capacity_ = 0;
    if (!cave_anchor_compute_available_ || cave_anchor_ssbo_ == 0 || cave_anchor_vbo_ == 0 || mesh.cave_anchor_points.empty()) {
        return;
    }

    std::vector<GpuCaveAnchor> gpu_anchors;
    gpu_anchors.reserve(mesh.cave_anchor_points.size());
    constexpr float CaveMouthCullRadiusKm = 22.0f;
    for (Vec3 anchor : mesh.cave_anchor_points) {
        const Vec3 direction = normalize(anchor);
        GpuCaveAnchor gpu_anchor;
        gpu_anchor.direction_radius[0] = direction.x;
        gpu_anchor.direction_radius[1] = direction.y;
        gpu_anchor.direction_radius[2] = direction.z;
        gpu_anchor.direction_radius[3] = CaveMouthCullRadiusKm;
        gpu_anchors.push_back(gpu_anchor);
    }

    cave_anchor_compute_count_ = static_cast<uint32_t>(gpu_anchors.size());
    cave_anchor_compute_capacity_ = std::min(cave_anchor_compute_count_, CaveAnchorComputeMaxVisible);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cave_anchor_ssbo_);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        static_cast<GLsizeiptr>(gpu_anchors.size() * sizeof(GpuCaveAnchor)),
        gpu_anchors.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ARRAY_BUFFER, cave_anchor_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(static_cast<uint64_t>(cave_anchor_compute_capacity_) * sizeof(DebugVertex)),
        nullptr,
        GL_DYNAMIC_DRAW
    );
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

bool DebugRenderer::render_cave_anchor_compute(const CameraView& view, float aspect, float far_plane) {
    if (!cave_anchor_compute_enabled_ ||
        !cave_anchor_compute_available_ ||
        cave_anchor_compute_shader_ == 0 ||
        cave_anchor_compute_count_ == 0u ||
        cave_anchor_compute_capacity_ == 0u ||
        cave_anchor_ssbo_ == 0 ||
        cave_anchor_counter_ssbo_ == 0 ||
        cave_anchor_indirect_buffer_ == 0 ||
        cave_anchor_vbo_ == 0) {
        return false;
    }

    const VoxelDebugView view_data = make_voxel_debug_view(view, aspect, far_plane);
    const uint32_t zero_counter = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cave_anchor_counter_ssbo_);
    glClearBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        GL_R32UI,
        0,
        sizeof(uint32_t),
        GL_RED_INTEGER,
        GL_UNSIGNED_INT,
        &zero_counter
    );
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cave_anchor_indirect_buffer_);
    glClearBufferSubData(
        GL_DRAW_INDIRECT_BUFFER,
        GL_R32UI,
        0,
        sizeof(uint32_t),
        GL_RED_INTEGER,
        GL_UNSIGNED_INT,
        &zero_counter
    );

    glUseProgram(cave_anchor_compute_shader_);
    glUniform1ui(cave_anchor_compute_anchor_count_location_, cave_anchor_compute_count_);
    glUniform1ui(cave_anchor_compute_max_visible_location_, cave_anchor_compute_capacity_);
    glUniform1f(cave_anchor_compute_planet_radius_location_, PlanetRadiusKilometers);
    glUniform3f(cave_anchor_compute_eye_location_, view_data.eye.x, view_data.eye.y, view_data.eye.z);
    glUniform3f(cave_anchor_compute_forward_location_, view_data.forward.x, view_data.forward.y, view_data.forward.z);
    glUniform3f(cave_anchor_compute_right_location_, view_data.right.x, view_data.right.y, view_data.right.z);
    glUniform3f(cave_anchor_compute_up_location_, view_data.up.x, view_data.up.y, view_data.up.z);
    glUniform1f(cave_anchor_compute_tan_half_fov_y_location_, view_data.tan_half_fov_y);
    glUniform1f(cave_anchor_compute_tan_half_fov_x_location_, view_data.tan_half_fov_x);
    glUniform1f(cave_anchor_compute_near_plane_location_, view_data.near_plane);
    glUniform1f(cave_anchor_compute_far_plane_location_, view_data.far_plane);
    glUniform1f(cave_anchor_compute_occlusion_radius_location_, view_data.occlusion_radius);
    glUniform1f(cave_anchor_compute_max_distance_location_, 900.0f);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cave_anchor_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, cave_anchor_vbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, cave_anchor_counter_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, cave_anchor_indirect_buffer_);

    const uint32_t group_count = (cave_anchor_compute_count_ + CaveAnchorComputeWorkgroupSize - 1u) / CaveAnchorComputeWorkgroupSize;
    glDispatchCompute(group_count, 1u, 1u);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    glUseProgram(shader_);
    glUniform1f(shader_point_size_location_, 5.0f);
    glUniform1i(shader_point_style_location_, 1);
    glBindVertexArray(cave_anchor_vao_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cave_anchor_indirect_buffer_);
    glDrawArraysIndirect(GL_POINTS, nullptr);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    ++perf_stats_.draw_calls;
    glUniform1i(shader_point_style_location_, 0);
    return true;
}

bool DebugRenderer::render_voxel_compute(const CameraView& view, float aspect, float far_plane) {
    if (!voxel_compute_enabled_ || !voxel_compute_available_ || voxel_compute_shader_ == 0) {
        perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackUnavailable;
        return false;
    }
    if (voxel_compute_node_count_ == 0u || voxel_compute_max_boxes_ == 0u || voxel_compute_vertex_capacity_ == 0u) {
        perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackWaitingForSvo;
        return false;
    }
    if (voxel_node_ssbo_ == 0 || voxel_counter_ssbo_ == 0 || voxel_indirect_buffer_ == 0 || voxel_vbo_ == 0) {
        perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackInvalidBuffers;
        return false;
    }

    const auto dispatch_begin = std::chrono::steady_clock::now();
    const VoxelDebugView voxel_view = make_voxel_debug_view(view, aspect, far_plane);

    const SparseVoxelOctreeNode& root_node = current_svo_.nodes.front();
    const Vec3 root_center = planet_to_world(svo_node_center(current_svo_, root_node));
    const float root_size = svo_node_world_size(current_svo_, root_node) * PlanetRadiusKilometers;
    if (!voxel_box_visible(root_center, root_size * 0.8661f, voxel_view)) {
        perf_stats_.voxel_compute_dispatch_ms = 0.0;
        perf_stats_.voxel_dynamic_upload_bytes = 0;
        perf_stats_.voxel_debug_lines = 0;
        perf_stats_.voxel_compute_available = voxel_compute_available_;
        perf_stats_.voxel_compute_used = true;
        perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackNone;
        return true;
    }

    const uint32_t zero_counter = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, voxel_counter_ssbo_);
    glClearBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        GL_R32UI,
        0,
        sizeof(uint32_t),
        GL_RED_INTEGER,
        GL_UNSIGNED_INT,
        &zero_counter
    );
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, voxel_indirect_buffer_);
    glClearBufferSubData(
        GL_DRAW_INDIRECT_BUFFER,
        GL_R32UI,
        0,
        sizeof(uint32_t),
        GL_RED_INTEGER,
        GL_UNSIGNED_INT,
        &zero_counter
    );

    glUseProgram(voxel_compute_shader_);
    glUniform1ui(voxel_compute_node_count_location_, voxel_compute_node_count_);
    glUniform1f(voxel_compute_bounds_radius_location_, current_svo_.bounds_radius);
    glUniform1ui(voxel_compute_svo_depth_location_, current_svo_.depth);
    glUniform1ui(voxel_compute_max_boxes_location_, voxel_compute_max_boxes_);
    glUniform1f(voxel_compute_planet_radius_location_, PlanetRadiusKilometers);
    glUniform3f(voxel_compute_eye_location_, voxel_view.eye.x, voxel_view.eye.y, voxel_view.eye.z);
    glUniform3f(voxel_compute_range_center_location_, voxel_view.range_center.x, voxel_view.range_center.y, voxel_view.range_center.z);
    glUniform3f(voxel_compute_forward_location_, voxel_view.forward.x, voxel_view.forward.y, voxel_view.forward.z);
    glUniform3f(voxel_compute_right_location_, voxel_view.right.x, voxel_view.right.y, voxel_view.right.z);
    glUniform3f(voxel_compute_up_location_, voxel_view.up.x, voxel_view.up.y, voxel_view.up.z);
    glUniform1f(voxel_compute_tan_half_fov_y_location_, voxel_view.tan_half_fov_y);
    glUniform1f(voxel_compute_tan_half_fov_x_location_, voxel_view.tan_half_fov_x);
    glUniform1f(voxel_compute_near_plane_location_, voxel_view.near_plane);
    glUniform1f(voxel_compute_far_plane_location_, voxel_view.far_plane);
    glUniform1f(voxel_compute_occlusion_radius_location_, voxel_view.occlusion_radius);
    glUniform1f(voxel_compute_range_radius_location_, voxel_view.range_radius);
    glUniform1f(voxel_compute_debug_box_size_location_, voxel_view.debug_box_size);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, voxel_node_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, voxel_vbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, voxel_counter_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, voxel_indirect_buffer_);

    const uint32_t group_count = (voxel_compute_node_count_ + VoxelComputeWorkgroupSize - 1u) / VoxelComputeWorkgroupSize;
    glDispatchCompute(group_count, 1u, 1u);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    glUseProgram(shader_);
    glUniform1f(shader_point_size_location_, 1.0f);
    glBindVertexArray(voxel_vao_);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, voxel_indirect_buffer_);
    glDrawArraysIndirect(GL_LINES, nullptr);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    ++perf_stats_.draw_calls;

    perf_stats_.voxel_compute_dispatch_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - dispatch_begin
    ).count();
    perf_stats_.voxel_dynamic_upload_bytes = 0;
    perf_stats_.voxel_debug_lines = 0;
    perf_stats_.voxel_compute_available = voxel_compute_available_;
    perf_stats_.voxel_compute_used = true;
    perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackNone;
    return true;
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
    shader_mvp_location_ = glGetUniformLocation(shader_, "u_mvp");
    shader_point_size_location_ = glGetUniformLocation(shader_, "u_point_size");
    shader_point_style_location_ = glGetUniformLocation(shader_, "u_point_style");
    shader_terrain_hole_count_location_ = glGetUniformLocation(shader_, "u_terrain_hole_count");
    shader_terrain_holes_location_ = glGetUniformLocation(shader_, "u_terrain_holes[0]");
    shader_terrain_mask_count_location_ = glGetUniformLocation(shader_, "u_terrain_mask_count");
    shader_terrain_mask_sampler_location_ = glGetUniformLocation(shader_, "u_terrain_height_masks");
    shader_terrain_mask_center_radius_location_ = glGetUniformLocation(shader_, "u_terrain_mask_center_radius[0]");
    shader_terrain_mask_tangent_location_ = glGetUniformLocation(shader_, "u_terrain_mask_tangent[0]");
    shader_terrain_mask_bitangent_location_ = glGetUniformLocation(shader_, "u_terrain_mask_bitangent[0]");
    surface_mvp_location_ = glGetUniformLocation(surface_net_shader_, "u_mvp");
    surface_camera_location_ = glGetUniformLocation(surface_net_shader_, "u_camera_position");
    surface_light_location_ = glGetUniformLocation(surface_net_shader_, "u_light_direction");
    initialize_voxel_compute();
    initialize_cave_anchor_compute();

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
    overlay_buffer_capacity_bytes_ = OverlayDefaultVertexCapacity * static_cast<uint32_t>(sizeof(DebugVertex));
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(overlay_buffer_capacity_bytes_), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(sizeof(Vec3)));
    glBindVertexArray(0);

    if (gpu_query_mesh_ == 0) {
        glGenQueries(1, &gpu_query_mesh_);
        glGenQueries(1, &gpu_query_surface_net_);
        glGenQueries(1, &gpu_query_debug_);
    }

    line_vertex_count_ = static_cast<int>(line_vertices.size());
    point_vertex_count_ = static_cast<int>(point_vertices.size());
    grid_ribbon_vertex_count_ = static_cast<int>(grid_ribbon_vertices.size());
    grid_ribbon_line_vertex_count_ = static_cast<int>(grid_ribbon_line_vertices.size());

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.015f, 0.018f, 0.024f, 1.0f);
    perf_stats_.voxel_compute_available = voxel_compute_available_;
    return true;
}

void DebugRenderer::update_mesh(const QuantizedMesh& mesh) {
    const auto upload_begin = std::chrono::steady_clock::now();
    release_mesh_buffers();

    const std::vector<DebugVertex> mesh_vertices = build_debug_mesh_vertices(mesh);
    const std::vector<DebugVertex> voxel_vertices;
    const std::vector<DebugVertex> cave_anchor_vertices = build_cave_anchor_vertices(mesh);
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
    upload_voxel_compute_buffers(mesh.svo);
    upload_vertex_buffer(cave_anchor_vao_, cave_anchor_vbo_, cave_anchor_vertices);
    upload_cave_anchor_compute_buffers(mesh);
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
    cave_anchor_vertex_count_ = static_cast<int>(cave_anchor_vertices.size());
    surface_net_index_count_ = static_cast<int>(mesh.surface_net.triangle_indices.size());
    surface_net_auto_visible_ = mesh.surface_net.local_patch_count > 0u;

    uint64_t mesh_upload_bytes = mesh_vertices.size() * sizeof(DebugVertex);
    mesh_upload_bytes += mesh.triangle_indices.size() * sizeof(uint32_t);
    mesh_upload_bytes += mesh.line_indices.size() * sizeof(uint32_t);
    mesh_upload_bytes += mesh.stitch_triangle_indices.size() * sizeof(uint32_t);
    mesh_upload_bytes += mesh.stitch_line_indices.size() * sizeof(uint32_t);

    uint64_t surface_upload_bytes = surface_net_vertices.size() * sizeof(SurfaceNetVertex);
    surface_upload_bytes += mesh.surface_net.triangle_indices.size() * sizeof(uint32_t);
    surface_upload_bytes += cave_anchor_vertices.size() * sizeof(DebugVertex);

    RendererPerfStats next_stats = perf_stats_;
    next_stats.upload_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - upload_begin
    ).count();
    next_stats.mesh_upload_bytes = mesh_upload_bytes;
    next_stats.surface_net_upload_bytes = surface_upload_bytes;
    next_stats.voxel_dynamic_upload_bytes = 0;
    next_stats.mesh_vertices = static_cast<uint32_t>(mesh.vertices.size());
    next_stats.mesh_triangles = static_cast<uint32_t>(mesh.triangle_indices.size() / 3u);
    next_stats.stitch_triangles = static_cast<uint32_t>(mesh.stitch_triangle_indices.size() / 3u);
    next_stats.mesh_lines = static_cast<uint32_t>(mesh.line_indices.size() / 2u);
    next_stats.stitch_lines = static_cast<uint32_t>(mesh.stitch_line_indices.size() / 2u);
    next_stats.surface_net_vertices = static_cast<uint32_t>(mesh.surface_net.vertices.size());
    next_stats.surface_net_triangles = static_cast<uint32_t>(mesh.surface_net.triangle_indices.size() / 3u);
    next_stats.cave_interior_vertices = 0;
    next_stats.cave_interior_triangles = 0;
    next_stats.voxel_debug_lines = 0;
    next_stats.voxel_compute_dispatch_ms = 0.0;
    next_stats.voxel_compute_available = voxel_compute_available_;
    next_stats.voxel_compute_used = false;
    next_stats.voxel_compute_fallback_code = voxel_compute_node_count_ > 0u
        ? VoxelComputeFallbackNone
        : (voxel_compute_available_ ? VoxelComputeFallbackWaitingForSvo : VoxelComputeFallbackUnavailable);
    perf_stats_ = next_stats;
}

void DebugRenderer::update_cave_interiors(const SurfaceNetMesh& cave_interiors) {
    const auto upload_begin = std::chrono::steady_clock::now();
    const std::vector<SurfaceNetVertex> cave_vertices = build_surface_net_vertices(cave_interiors);
    if (cave_interior_vao_ == 0) {
        glGenVertexArrays(1, &cave_interior_vao_);
    }
    if (cave_interior_vbo_ == 0) {
        glGenBuffers(1, &cave_interior_vbo_);
    }
    if (cave_interior_ebo_ == 0) {
        glGenBuffers(1, &cave_interior_ebo_);
    }

    glBindVertexArray(cave_interior_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, cave_interior_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(cave_vertices.size() * sizeof(SurfaceNetVertex)),
        cave_vertices.empty() ? nullptr : cave_vertices.data(),
        GL_DYNAMIC_DRAW
    );
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceNetVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceNetVertex), reinterpret_cast<void*>(sizeof(Vec3)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(SurfaceNetVertex), reinterpret_cast<void*>(sizeof(Vec3) * 2));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cave_interior_ebo_);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(cave_interiors.triangle_indices.size() * sizeof(uint32_t)),
        cave_interiors.triangle_indices.empty() ? nullptr : cave_interiors.triangle_indices.data(),
        GL_DYNAMIC_DRAW
    );
    glBindVertexArray(0);

    cave_interior_index_count_ = static_cast<int>(cave_interiors.triangle_indices.size());
    perf_stats_.cave_interior_vertices = static_cast<uint32_t>(cave_interiors.vertices.size());
    perf_stats_.cave_interior_triangles = static_cast<uint32_t>(cave_interiors.triangle_indices.size() / 3u);
    perf_stats_.surface_net_upload_bytes += cave_vertices.size() * sizeof(SurfaceNetVertex);
    perf_stats_.surface_net_upload_bytes += cave_interiors.triangle_indices.size() * sizeof(uint32_t);
    perf_stats_.upload_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - upload_begin
    ).count();
}

void DebugRenderer::update_terrain_holes(const std::vector<LocalVoxelFeature>& holes) {
    terrain_holes_.clear();
    terrain_holes_.reserve(std::min<size_t>(holes.size(), MaxTerrainHoles));
    for (const LocalVoxelFeature& hole : holes) {
        if (hole.kind != VoxelFeatureKind::CaveSystem || hole.entrance_radius_km <= 0.0f || length(hole.center_mesh) <= 0.000001f) {
            continue;
        }
        terrain_holes_.push_back(hole);
        if (terrain_holes_.size() >= MaxTerrainHoles) {
            break;
        }
    }
    rebuild_terrain_transition_buffer();
}

void DebugRenderer::rebuild_terrain_transition_buffer() {
    std::vector<DebugVertex> transition_vertices = build_cave_transition_vertices(terrain_holes_);
    const std::vector<DebugVertex> mask_vertices = build_terrain_mask_transition_vertices(terrain_masks_);
    transition_vertices.insert(transition_vertices.end(), mask_vertices.begin(), mask_vertices.end());
    if (cave_transition_vao_ == 0) {
        glGenVertexArrays(1, &cave_transition_vao_);
    }
    if (cave_transition_vbo_ == 0) {
        glGenBuffers(1, &cave_transition_vbo_);
    }
    glBindVertexArray(cave_transition_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, cave_transition_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(transition_vertices.size() * sizeof(DebugVertex)),
        transition_vertices.empty() ? nullptr : transition_vertices.data(),
        GL_DYNAMIC_DRAW
    );
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(DebugVertex), reinterpret_cast<void*>(sizeof(Vec3)));
    glBindVertexArray(0);
    cave_transition_vertex_count_ = static_cast<int>(transition_vertices.size());
}

void DebugRenderer::update_terrain_height_masks(const std::vector<TerrainHeightMask>& masks) {
    terrain_masks_.clear();
    terrain_masks_.reserve(std::min<size_t>(masks.size(), MaxTerrainHeightMasks));
    for (const TerrainHeightMask& mask : masks) {
        if (mask.resolution == 0u ||
            mask.radius_km <= 0.0f ||
            mask.heights.size() != static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution) ||
            length(mask.center_mesh) <= 0.000001f) {
            continue;
        }
        terrain_masks_.push_back(mask);
        if (terrain_masks_.size() >= MaxTerrainHeightMasks) {
            break;
        }
    }

    const uint32_t mask_count = static_cast<uint32_t>(terrain_masks_.size());
    const uint32_t resolution = mask_count == 0u ? 0u : terrain_masks_.front().resolution;
    if (mask_count == 0u || resolution == 0u) {
        terrain_mask_resolution_ = 0u;
        terrain_mask_layer_count_ = 0u;
        rebuild_terrain_transition_buffer();
        return;
    }

    std::vector<uint16_t> texture_data(static_cast<size_t>(resolution) * resolution * mask_count, 8191u);
    uint32_t layer_count = 0u;
    for (const TerrainHeightMask& mask : terrain_masks_) {
        if (mask.resolution != resolution) {
            continue;
        }
        const size_t layer_offset = static_cast<size_t>(layer_count) * resolution * resolution;
        std::copy(mask.heights.begin(), mask.heights.end(), texture_data.begin() + static_cast<std::ptrdiff_t>(layer_offset));
        ++layer_count;
    }
    if (layer_count == 0u) {
        terrain_mask_resolution_ = 0u;
        terrain_mask_layer_count_ = 0u;
        rebuild_terrain_transition_buffer();
        return;
    }
    texture_data.resize(static_cast<size_t>(resolution) * resolution * layer_count);

    if (terrain_mask_texture_ == 0) {
        glGenTextures(1, &terrain_mask_texture_);
    }
    glBindTexture(GL_TEXTURE_2D_ARRAY, terrain_mask_texture_);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(
        GL_TEXTURE_2D_ARRAY,
        0,
        GL_R16UI,
        static_cast<GLsizei>(resolution),
        static_cast<GLsizei>(resolution),
        static_cast<GLsizei>(layer_count),
        0,
        GL_RED_INTEGER,
        GL_UNSIGNED_SHORT,
        texture_data.data()
    );
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    terrain_mask_resolution_ = resolution;
    terrain_mask_layer_count_ = layer_count;
    rebuild_terrain_transition_buffer();
}

void DebugRenderer::resize(int width, int height) {
    width_ = width > 0 ? width : 1;
    height_ = height > 0 ? height : 1;
}

void DebugRenderer::render(const CameraView& view, const SpaceshipState& ship, const DebugRenderOptions& options, bool show_fps, float fps) {
    const auto render_begin = std::chrono::steady_clock::now();
    const bool sample_gpu_timers = show_fps && ((gpu_timer_frame_++ & 15u) == 0u);
    if (show_fps) {
        read_gpu_timer(gpu_query_mesh_, perf_stats_.gpu_mesh_ms);
        read_gpu_timer(gpu_query_surface_net_, perf_stats_.gpu_surface_net_ms);
        read_gpu_timer(gpu_query_debug_, perf_stats_.gpu_debug_ms);
    }
    perf_stats_.draw_calls = 0;

    glViewport(0, 0, width_, height_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float aspect = static_cast<float>(width_) / static_cast<float>(height_);
    const float far_plane = std::max(PlanetRadiusKilometers * 16.0f, length(view.eye) + PlanetRadiusKilometers * 4.0f);
    const Mat4 projection = perspective(50.0f * Pi / 180.0f, aspect, 0.05f, far_plane);
    const Mat4 view_matrix = look_at(view.eye, view.target, view.up);
    const Mat4 mvp = projection * view_matrix;

    glUseProgram(shader_);
    glUniformMatrix4fv(shader_mvp_location_, 1, GL_FALSE, mvp.m);

    glUniform1f(shader_point_size_location_, 1.0f);
    glUniform1i(shader_point_style_location_, 0);
    if (shader_terrain_mask_count_location_ >= 0) {
        const int mask_count = static_cast<int>(std::min<uint32_t>(terrain_mask_layer_count_, MaxTerrainHeightMasks));
        glUniform1i(shader_terrain_mask_count_location_, mask_count);
        if (mask_count > 0 && terrain_mask_texture_ != 0) {
            std::array<float, MaxTerrainHeightMasks * 4u> centers = {};
            std::array<float, MaxTerrainHeightMasks * 4u> tangents = {};
            std::array<float, MaxTerrainHeightMasks * 4u> bitangents = {};
            for (int i = 0; i < mask_count; ++i) {
                const TerrainHeightMask& mask = terrain_masks_[static_cast<size_t>(i)];
                const Vec3 center = normalize(mask.center_mesh);
                const Vec3 tangent = normalize(mask.tangent_mesh);
                const Vec3 bitangent = normalize(mask.bitangent_mesh);
                centers[static_cast<size_t>(i) * 4u + 0u] = center.x;
                centers[static_cast<size_t>(i) * 4u + 1u] = center.y;
                centers[static_cast<size_t>(i) * 4u + 2u] = center.z;
                centers[static_cast<size_t>(i) * 4u + 3u] = kilometers_to_world_units(mask.radius_km);
                tangents[static_cast<size_t>(i) * 4u + 0u] = tangent.x;
                tangents[static_cast<size_t>(i) * 4u + 1u] = tangent.y;
                tangents[static_cast<size_t>(i) * 4u + 2u] = tangent.z;
                bitangents[static_cast<size_t>(i) * 4u + 0u] = bitangent.x;
                bitangents[static_cast<size_t>(i) * 4u + 1u] = bitangent.y;
                bitangents[static_cast<size_t>(i) * 4u + 2u] = bitangent.z;
            }
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D_ARRAY, terrain_mask_texture_);
            if (shader_terrain_mask_sampler_location_ >= 0) {
                glUniform1i(shader_terrain_mask_sampler_location_, 0);
            }
            if (shader_terrain_mask_center_radius_location_ >= 0) {
                glUniform4fv(shader_terrain_mask_center_radius_location_, mask_count, centers.data());
            }
            if (shader_terrain_mask_tangent_location_ >= 0) {
                glUniform4fv(shader_terrain_mask_tangent_location_, mask_count, tangents.data());
            }
            if (shader_terrain_mask_bitangent_location_ >= 0) {
                glUniform4fv(shader_terrain_mask_bitangent_location_, mask_count, bitangents.data());
            }
        }
    }
    if (shader_terrain_hole_count_location_ >= 0) {
        const int hole_count = static_cast<int>(std::min<size_t>(terrain_holes_.size(), MaxTerrainHoles));
        glUniform1i(shader_terrain_hole_count_location_, hole_count);
        if (hole_count > 0 && shader_terrain_holes_location_ >= 0) {
            std::array<float, MaxTerrainHoles * 4u> hole_data = {};
            for (int i = 0; i < hole_count; ++i) {
                const LocalVoxelFeature& hole = terrain_holes_[static_cast<size_t>(i)];
                const Vec3 direction = normalize(hole.center_mesh);
                const float chord_radius = hole.entrance_radius_km / PlanetRadiusKilometers;
                hole_data[static_cast<size_t>(i) * 4u + 0u] = direction.x;
                hole_data[static_cast<size_t>(i) * 4u + 1u] = direction.y;
                hole_data[static_cast<size_t>(i) * 4u + 2u] = direction.z;
                hole_data[static_cast<size_t>(i) * 4u + 3u] = chord_radius;
            }
            glUniform4fv(shader_terrain_holes_location_, hole_count, hole_data.data());
        }
    }
    glBindVertexArray(mesh_vao_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_triangle_ebo_);
    if (sample_gpu_timers) {
        begin_gpu_timer(gpu_query_mesh_);
    }
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
    glDrawElements(GL_TRIANGLES, mesh_triangle_index_count_, GL_UNSIGNED_INT, nullptr);
    ++perf_stats_.draw_calls;
    glDisable(GL_POLYGON_OFFSET_FILL);

    if (options.show_mesh_wire) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_line_ebo_);
        glDisableVertexAttribArray(1);
        glVertexAttrib3f(1, 0.42f, 0.92f, 1.0f);
        glDrawElements(GL_LINES, mesh_line_index_count_, GL_UNSIGNED_INT, nullptr);
        ++perf_stats_.draw_calls;
        glEnableVertexAttribArray(1);
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, stitch_triangle_ebo_);
    glDrawElements(GL_TRIANGLES, stitch_triangle_index_count_, GL_UNSIGNED_INT, nullptr);
    ++perf_stats_.draw_calls;
    if (sample_gpu_timers) {
        end_gpu_timer();
    }
    if (shader_terrain_hole_count_location_ >= 0) {
        glUniform1i(shader_terrain_hole_count_location_, 0);
    }
    if (shader_terrain_mask_count_location_ >= 0) {
        glUniform1i(shader_terrain_mask_count_location_, 0);
    }
    if (cave_transition_vertex_count_ > 0) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-0.75f, -0.75f);
        glBindVertexArray(cave_transition_vao_);
        glDrawArrays(GL_TRIANGLES, 0, cave_transition_vertex_count_);
        ++perf_stats_.draw_calls;
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    const bool render_surface_net = options.show_surface_net || surface_net_auto_visible_;
    if (render_surface_net || cave_interior_index_count_ > 0) {
        if (sample_gpu_timers) {
            begin_gpu_timer(gpu_query_surface_net_);
        }
        glUseProgram(surface_net_shader_);
        const Vec3 light_direction = normalize(Vec3{-0.35f, -0.75f, -0.50f});
        glUniformMatrix4fv(surface_mvp_location_, 1, GL_FALSE, mvp.m);
        glUniform3f(surface_camera_location_, view.eye.x, view.eye.y, view.eye.z);
        glUniform3f(surface_light_location_, light_direction.x, light_direction.y, light_direction.z);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.0f, -1.0f);
        if (render_surface_net && surface_net_index_count_ > 0) {
            glBindVertexArray(surface_net_vao_);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, surface_net_ebo_);
            glDrawElements(GL_TRIANGLES, surface_net_index_count_, GL_UNSIGNED_INT, nullptr);
            ++perf_stats_.draw_calls;
        }
        if (cave_interior_index_count_ > 0) {
            glBindVertexArray(cave_interior_vao_);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cave_interior_ebo_);
            glDrawElements(GL_TRIANGLES, cave_interior_index_count_, GL_UNSIGNED_INT, nullptr);
            ++perf_stats_.draw_calls;
        }
        glDisable(GL_POLYGON_OFFSET_FILL);
        glUseProgram(shader_);
        glUniformMatrix4fv(shader_mvp_location_, 1, GL_FALSE, mvp.m);
        glUniform1f(shader_point_size_location_, 1.0f);
        if (sample_gpu_timers) {
            end_gpu_timer();
        }
    }

    if (sample_gpu_timers) {
        begin_gpu_timer(gpu_query_debug_);
    }
    if (options.show_goldberg_grid) {
        glDepthFunc(GL_LEQUAL);
        glBindVertexArray(grid_ribbon_vao_);
        glDrawArrays(GL_TRIANGLES, 0, grid_ribbon_vertex_count_);
        ++perf_stats_.draw_calls;

        glBindVertexArray(grid_ribbon_line_vao_);
        glDrawArrays(GL_LINES, 0, grid_ribbon_line_vertex_count_);
        ++perf_stats_.draw_calls;
        glDepthFunc(GL_LESS);

        glUniform1f(shader_point_size_location_, 1.0f);
        glBindVertexArray(line_vao_);
        glDrawArrays(GL_LINES, 0, line_vertex_count_);
        ++perf_stats_.draw_calls;
    }

    if (options.show_points) {
        glUniform1f(shader_point_size_location_, 6.0f);
        glBindVertexArray(point_vao_);
        glDrawArrays(GL_POINTS, 0, point_vertex_count_);
        ++perf_stats_.draw_calls;
    }

    if (options.show_cave_anchors) {
        if (!render_cave_anchor_compute(view, aspect, far_plane)) {
            glUniform1f(shader_point_size_location_, 5.0f);
            glUniform1i(shader_point_style_location_, 1);
            glBindVertexArray(cave_anchor_vao_);
            glDrawArrays(GL_POINTS, 0, cave_anchor_vertex_count_);
            ++perf_stats_.draw_calls;
            glUniform1i(shader_point_style_location_, 0);
        }
    }

    if (options.show_voxels) {
        if (!render_voxel_compute(view, aspect, far_plane)) {
            const std::vector<DebugVertex> voxel_vertices = build_visible_voxel_vertices(current_svo_, view, aspect, far_plane);
            voxel_line_vertex_count_ = static_cast<int>(voxel_vertices.size());
            perf_stats_.voxel_dynamic_upload_bytes = voxel_vertices.size() * sizeof(DebugVertex);
            perf_stats_.voxel_debug_lines = static_cast<uint32_t>(voxel_vertices.size() / 2u);
            perf_stats_.voxel_compute_dispatch_ms = 0.0;
            perf_stats_.voxel_compute_available = voxel_compute_available_;
            perf_stats_.voxel_compute_used = false;
            if (perf_stats_.voxel_compute_fallback_code == VoxelComputeFallbackNone) {
                perf_stats_.voxel_compute_fallback_code = voxel_compute_available_
                    ? VoxelComputeFallbackInvalidBuffers
                    : VoxelComputeFallbackUnavailable;
            }
            glBindVertexArray(voxel_vao_);
            glBindBuffer(GL_ARRAY_BUFFER, voxel_vbo_);
            glBufferData(
                GL_ARRAY_BUFFER,
                static_cast<GLsizeiptr>(voxel_vertices.size() * sizeof(DebugVertex)),
                voxel_vertices.data(),
                GL_DYNAMIC_DRAW
            );
            glDrawArrays(GL_LINES, 0, voxel_line_vertex_count_);
            ++perf_stats_.draw_calls;
        }
    } else {
        perf_stats_.voxel_dynamic_upload_bytes = 0;
        perf_stats_.voxel_debug_lines = 0;
        perf_stats_.voxel_compute_dispatch_ms = 0.0;
        perf_stats_.voxel_compute_available = voxel_compute_available_;
        perf_stats_.voxel_compute_used = false;
        perf_stats_.voxel_compute_fallback_code = VoxelComputeFallbackNone;
    }

    std::array<DebugVertex, (SpaceshipState::TrailCapacity - 1u) * 2u + 18u> overlay_vertices = {};
    const uint32_t trail_vertex_count = build_spaceship_trail_vertices(ship, overlay_vertices.data());
    const uint32_t ship_vertex_count = build_spaceship_vertices(ship, overlay_vertices.data() + trail_vertex_count);
    const uint32_t overlay_vertex_count = trail_vertex_count + ship_vertex_count;
    glUniform1f(shader_point_size_location_, 1.0f);
    glUniform1i(shader_point_style_location_, 0);
    glBindVertexArray(overlay_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo_);
    upload_dynamic_overlay_vertices(overlay_buffer_capacity_bytes_, overlay_vertices.data(), overlay_vertex_count);
    if (trail_vertex_count > 0u) {
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(trail_vertex_count));
        ++perf_stats_.draw_calls;
    }
    glDrawArrays(GL_TRIANGLES, static_cast<GLint>(trail_vertex_count), static_cast<GLsizei>(ship_vertex_count));
    ++perf_stats_.draw_calls;
    if (sample_gpu_timers) {
        end_gpu_timer();
    }

    glBindVertexArray(0);
    glUseProgram(0);

    if (show_fps) {
        render_fps_overlay(fps);
    }
    perf_stats_.render_cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - render_begin
    ).count();
}

void DebugRenderer::begin_gpu_timer(uint32_t query) {
    if (query != 0 && GLAD_GL_VERSION_3_3) {
        glBeginQuery(GL_TIME_ELAPSED, query);
    }
}

void DebugRenderer::end_gpu_timer() {
    if (GLAD_GL_VERSION_3_3) {
        glEndQuery(GL_TIME_ELAPSED);
    }
}

void DebugRenderer::read_gpu_timer(uint32_t query, double& milliseconds) {
    if (query == 0 || !GLAD_GL_VERSION_3_3) {
        milliseconds = 0.0;
        return;
    }

    GLint available = 0;
    glGetQueryObjectiv(query, GL_QUERY_RESULT_AVAILABLE, &available);
    if (available == 0) {
        return;
    }

    GLuint64 nanoseconds = 0;
    glGetQueryObjectui64v(query, GL_QUERY_RESULT, &nanoseconds);
    milliseconds = static_cast<double>(nanoseconds) / 1000000.0;
}

void DebugRenderer::render_fps_overlay(float fps) {
    const std::vector<DebugVertex> vertices = build_fps_overlay_vertices(fps);

    glUseProgram(shader_);
    const Mat4 overlay_transform = identity();
    glUniformMatrix4fv(shader_mvp_location_, 1, GL_FALSE, overlay_transform.m);
    glUniform1f(shader_point_size_location_, 1.0f);
    glUniform1i(shader_point_style_location_, 0);

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(overlay_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo_);
    upload_dynamic_overlay_vertices(overlay_buffer_capacity_bytes_, vertices.data(), vertices.size());
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices.size()));
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(0);
}

void DebugRenderer::render_progress_overlay(double progress) {
    const std::vector<DebugVertex> vertices = build_progress_overlay_vertices(progress);

    glUseProgram(shader_);
    const Mat4 overlay_transform = identity();
    glUniformMatrix4fv(shader_mvp_location_, 1, GL_FALSE, overlay_transform.m);
    glUniform1f(shader_point_size_location_, 1.0f);
    glUniform1i(shader_point_style_location_, 0);

    glDisable(GL_DEPTH_TEST);
    glBindVertexArray(overlay_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo_);
    upload_dynamic_overlay_vertices(overlay_buffer_capacity_bytes_, vertices.data(), vertices.size());
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
    if (cave_anchor_vbo_ != 0) {
        glDeleteBuffers(1, &cave_anchor_vbo_);
        cave_anchor_vbo_ = 0;
    }
    if (cave_anchor_vao_ != 0) {
        glDeleteVertexArrays(1, &cave_anchor_vao_);
        cave_anchor_vao_ = 0;
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
    if (cave_interior_ebo_ != 0) {
        glDeleteBuffers(1, &cave_interior_ebo_);
        cave_interior_ebo_ = 0;
    }
    if (cave_interior_vbo_ != 0) {
        glDeleteBuffers(1, &cave_interior_vbo_);
        cave_interior_vbo_ = 0;
    }
    if (cave_interior_vao_ != 0) {
        glDeleteVertexArrays(1, &cave_interior_vao_);
        cave_interior_vao_ = 0;
    }
    if (cave_transition_vbo_ != 0) {
        glDeleteBuffers(1, &cave_transition_vbo_);
        cave_transition_vbo_ = 0;
    }
    if (cave_transition_vao_ != 0) {
        glDeleteVertexArrays(1, &cave_transition_vao_);
        cave_transition_vao_ = 0;
    }

    mesh_triangle_index_count_ = 0;
    mesh_line_index_count_ = 0;
    stitch_triangle_index_count_ = 0;
    stitch_line_index_count_ = 0;
    voxel_line_vertex_count_ = 0;
    cave_anchor_vertex_count_ = 0;
    cave_anchor_compute_count_ = 0;
    cave_anchor_compute_capacity_ = 0;
    surface_net_index_count_ = 0;
    cave_interior_index_count_ = 0;
    cave_transition_vertex_count_ = 0;
    voxel_compute_node_count_ = 0;
    voxel_compute_max_boxes_ = 0;
    voxel_compute_vertex_capacity_ = 0;
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
    if (voxel_indirect_buffer_ != 0) {
        glDeleteBuffers(1, &voxel_indirect_buffer_);
        voxel_indirect_buffer_ = 0;
    }
    if (voxel_counter_ssbo_ != 0) {
        glDeleteBuffers(1, &voxel_counter_ssbo_);
        voxel_counter_ssbo_ = 0;
    }
    if (voxel_node_ssbo_ != 0) {
        glDeleteBuffers(1, &voxel_node_ssbo_);
        voxel_node_ssbo_ = 0;
    }
    if (cave_anchor_indirect_buffer_ != 0) {
        glDeleteBuffers(1, &cave_anchor_indirect_buffer_);
        cave_anchor_indirect_buffer_ = 0;
    }
    if (cave_anchor_counter_ssbo_ != 0) {
        glDeleteBuffers(1, &cave_anchor_counter_ssbo_);
        cave_anchor_counter_ssbo_ = 0;
    }
    if (cave_anchor_ssbo_ != 0) {
        glDeleteBuffers(1, &cave_anchor_ssbo_);
        cave_anchor_ssbo_ = 0;
    }
    if (terrain_mask_texture_ != 0) {
        glDeleteTextures(1, &terrain_mask_texture_);
        terrain_mask_texture_ = 0;
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
    overlay_buffer_capacity_bytes_ = 0;
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
    if (voxel_compute_shader_ != 0) {
        glDeleteProgram(voxel_compute_shader_);
        voxel_compute_shader_ = 0;
    }
    if (cave_anchor_compute_shader_ != 0) {
        glDeleteProgram(cave_anchor_compute_shader_);
        cave_anchor_compute_shader_ = 0;
    }
    if (gpu_query_debug_ != 0) {
        glDeleteQueries(1, &gpu_query_debug_);
        gpu_query_debug_ = 0;
    }
    if (gpu_query_surface_net_ != 0) {
        glDeleteQueries(1, &gpu_query_surface_net_);
        gpu_query_surface_net_ = 0;
    }
    if (gpu_query_mesh_ != 0) {
        glDeleteQueries(1, &gpu_query_mesh_);
        gpu_query_mesh_ = 0;
    }
    line_vertex_count_ = 0;
    point_vertex_count_ = 0;
    grid_ribbon_vertex_count_ = 0;
    grid_ribbon_line_vertex_count_ = 0;
    voxel_compute_available_ = false;
    voxel_compute_enabled_ = false;
}

} // namespace ae
