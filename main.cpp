#include "aetheronus/marching_cubes_tables.hpp"
#include "aetheronus/math.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/topology.hpp"

#include <glad/glad.h>
#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr float PlanetRadius = 1.0f;
constexpr float GridRadius = 1.08f;
constexpr float CoreVoidRadius = 0.20f;

struct RenderVertex {
    ae::Vec3 position;
    ae::Vec3 normal;
    ae::Vec3 color;
};

struct MinePath {
    std::vector<ae::Vec3> points;
    float radius = 0.045f;
    bool primary = false;
};

struct MineChamber {
    ae::Vec3 center;
    float radius = 0.08f;
    bool resource = false;
};

struct MineEntrance {
    uint32_t sample_id = 0;
    ae::Vec3 direction;
};

struct MineNetwork {
    std::vector<MineEntrance> entrances;
    std::vector<MinePath> primary_shafts;
    std::vector<MinePath> branches;
    std::vector<MineChamber> chambers;
};

struct MineSettings {
    uint32_t seed = 1337u;
    uint32_t mine_density = 24u;
    float tunnel_radius = 0.046f;
    uint32_t requested_grid_resolution = 512u;
};

enum class SdfEditMode : uint8_t {
    Carve,
    Fill,
};

enum class SdfEditShape : uint8_t {
    Sphere,
    Capsule,
};

enum class EditMaterial : uint8_t {
    Auto,
    Crust,
    MineWall,
    Resource,
    Fill,
};

enum class EditableObjectKind : uint8_t {
    Chamber,
    Tunnel,
};

struct SdfEdit {
    uint32_t id = 0;
    uint32_t object_id = 0;
    SdfEditMode mode = SdfEditMode::Carve;
    SdfEditShape shape = SdfEditShape::Sphere;
    ae::Vec3 center;
    ae::Vec3 a;
    ae::Vec3 b;
    float radius = 0.05f;
    float blend = 0.020f;
    EditMaterial material = EditMaterial::MineWall;
};

struct EditableMineObject {
    uint32_t id = 0;
    EditableObjectKind kind = EditableObjectKind::Chamber;
    std::vector<ae::Vec3> points;
    float radius = 0.06f;
    EditMaterial material = EditMaterial::MineWall;
    bool enabled = true;
};

struct BuildStats {
    bool ok = false;
    uint32_t seed = 0;
    uint32_t entrances = 0;
    uint32_t primary_shafts = 0;
    uint32_t branch_tunnels = 0;
    uint32_t chambers = 0;
    uint32_t grid_resolution = 0;
    uint32_t vertices = 0;
    uint32_t triangles = 0;
    uint32_t invalid_indices = 0;
    uint32_t degenerate_triangles = 0;
    uint32_t open_edges = 0;
    uint32_t nonmanifold_edges = 0;
    uint32_t edit_count = 0;
    uint32_t dirty_chunks = 0;
    float brush_radius = 0.046f;
    double last_remesh_ms = 0.0;
    std::string message;
};

struct TestbedMesh {
    std::vector<RenderVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<RenderVertex> graph_lines;
    BuildStats stats;
};

struct GpuMesh {
    uint32_t line_vao = 0;
    uint32_t line_vbo = 0;
    uint32_t line_vertex_count = 0;
    uint32_t chunk_index_count = 0;
};

struct GpuChunk {
    uint32_t vao = 0;
    uint32_t vbo = 0;
    uint32_t ebo = 0;
    uint32_t index_count = 0;
    uint32_t generation = 0;
};

struct GpuChunkStore {
    std::unordered_map<uint64_t, GpuChunk> chunks;
    std::deque<uint64_t> urgent_uploads;
    std::deque<uint64_t> pending_uploads;
    std::unordered_set<uint64_t> urgent_codes;
    std::unordered_set<uint64_t> pending_codes;
};

struct MortonChunkKey {
    uint64_t code = 0;
    uint32_t lod = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
};

enum class ChunkState : uint8_t {
    Empty,
    Queued,
    Building,
    Ready,
};

struct MineSdfChunk {
    MortonChunkKey key;
    uint32_t min_x = 0;
    uint32_t min_y = 0;
    uint32_t min_z = 0;
    uint32_t max_x = 0;
    uint32_t max_y = 0;
    uint32_t max_z = 0;
    ChunkState state = ChunkState::Empty;
    bool dirty = true;
    bool queued = false;
    bool gpu_dirty = true;
    bool urgent = false;
    uint32_t build_serial = 1;
    float priority = 0.0f;
    std::vector<float> values;
    std::vector<uint8_t> material_tags;
    std::vector<RenderVertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t invalid_indices = 0;
    uint32_t degenerate_triangles = 0;
};

struct SdfPrimitive {
    SdfEditMode mode = SdfEditMode::Carve;
    SdfEditShape shape = SdfEditShape::Sphere;
    ae::Vec3 center;
    ae::Vec3 a;
    ae::Vec3 b;
    ae::Vec3 minimum;
    ae::Vec3 maximum;
    float radius = 0.05f;
    float blend = 0.02f;
    EditMaterial material = EditMaterial::MineWall;
};

struct ChunkBuildJob {
    uint32_t generation = 0;
    MortonChunkKey key;
    float priority = 0.0f;
};

struct ChunkBuildResult {
    uint32_t generation = 0;
    uint32_t build_serial = 0;
    MortonChunkKey key;
    MineSdfChunk chunk;
};

struct MineSdfField {
    uint32_t resolution = 0;
    uint32_t point_resolution = 0;
    uint32_t chunk_size = 16;
    uint32_t chunks_x = 0;
    uint32_t chunks_y = 0;
    uint32_t chunks_z = 0;
    float voxel = 0.0f;
    float brush_radius = 0.046f;
    double last_remesh_ms = 0.0;
    uint32_t generation = 1;
    bool streaming_paused = false;
    uint32_t next_edit_id = 1;
    uint32_t next_object_id = 1;
    std::vector<SdfPrimitive> procedural_primitives;
    std::vector<SdfPrimitive> edit_primitives;
    std::vector<MortonChunkKey> active_keys;
    std::unordered_map<uint64_t, MineSdfChunk> chunks;
    std::unordered_set<uint64_t> queued_codes;
    std::vector<SdfEdit> edits;
    std::vector<SdfEdit> redo_edits;
    std::vector<EditableMineObject> objects;
    bool has_tunnel_mark = false;
    ae::Vec3 tunnel_mark;
    uint32_t completed_jobs_total = 0;
    uint32_t completed_jobs_last_frame = 0;
    uint32_t queued_jobs_last_frame = 0;
    uint32_t uploaded_chunks_last_frame = 0;
    std::array<uint32_t, 3u> visible_lod_counts = {0u, 0u, 0u};
    uint32_t transition_triangle_count = 0;
    uint32_t stale_result_drops = 0;
    double fps_estimate = 0.0;
};

struct WorkerTask {
    uint32_t generation = 0;
    uint64_t sequence = 0;
    float priority = 0.0f;
    std::function<ChunkBuildResult()> build;
};

struct WorkerTaskCompare {
    bool operator()(const WorkerTask& lhs, const WorkerTask& rhs) const {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.sequence > rhs.sequence;
    }
};

struct WorkerPool {
    std::vector<std::thread> threads;
    std::priority_queue<WorkerTask, std::vector<WorkerTask>, WorkerTaskCompare> queue;
    std::vector<ChunkBuildResult> completed;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> stop = false;
    uint64_t next_sequence = 1;
    uint32_t worker_count = 0;
};

struct Camera {
    bool fly_mode = true;
    ae::Vec3 position = {0.0f, 0.0f, 0.78f};
    ae::Vec3 forward = {0.0f, 0.0f, -1.0f};
    ae::Vec3 right = {1.0f, 0.0f, 0.0f};
    ae::Vec3 up = {0.0f, 1.0f, 0.0f};
    float orbit_yaw = 3.8f;
    float orbit_pitch = -0.35f;
    float orbit_distance = 3.0f;
};

struct EdgeKey {
    uint32_t a = 0;
    uint32_t b = 0;

    bool operator==(EdgeKey rhs) const {
        return a == rhs.a && b == rhs.b;
    }
};

struct EdgeKeyHash {
    size_t operator()(EdgeKey key) const {
        return (static_cast<size_t>(key.a) << 32u) ^ static_cast<size_t>(key.b);
    }
};

struct VertexKey {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    bool operator==(VertexKey rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct VertexKeyHash {
    size_t operator()(VertexKey key) const {
        size_t h = static_cast<size_t>(key.x) * 73856093ull;
        h ^= static_cast<size_t>(key.y) * 19349663ull;
        h ^= static_cast<size_t>(key.z) * 83492791ull;
        return h;
    }
};

uint32_t hash_u32(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float hash_unit(uint32_t value) {
    return static_cast<float>(hash_u32(value) & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

float saturate(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

float lerp_float(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

float smooth_min(float a, float b, float k) {
    if (k <= 0.0f) {
        return std::min(a, b);
    }
    const float h = saturate(0.5f + 0.5f * (b - a) / k);
    return lerp_float(b, a, h) - k * h * (1.0f - h);
}

EdgeKey make_edge_key(uint32_t a, uint32_t b) {
    if (b < a) {
        std::swap(a, b);
    }
    return {a, b};
}

bool finite_vec3(ae::Vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

ae::Vec3 color_crust() { return {0.11f, 0.47f, 0.53f}; }
ae::Vec3 color_mine_wall() { return {0.12f, 0.14f, 0.15f}; }
ae::Vec3 color_resource() { return {1.0f, 0.55f, 0.12f}; }
ae::Vec3 color_primary_line() { return {1.0f, 0.82f, 0.20f}; }
ae::Vec3 color_branch_line() { return {0.95f, 0.35f, 0.10f}; }
ae::Vec3 color_chamber_line() { return {0.35f, 0.82f, 1.0f}; }
ae::Vec3 color_fill_line() { return {0.62f, 0.95f, 0.72f}; }
ae::Vec3 color_brush_line() { return {1.0f, 1.0f, 1.0f}; }

uint8_t material_tag(EditMaterial material) {
    return static_cast<uint8_t>(material);
}

EditMaterial tag_material(uint8_t tag) {
    return static_cast<EditMaterial>(tag);
}

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
    const char* vertex_source = R"GLSL(
#version 430 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_position = world.xyz;
    v_normal = mat3(u_model) * a_normal;
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
)GLSL";

    const char* fragment_source = R"GLSL(
#version 430 core
in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
uniform vec3 u_camera_position;
uniform bool u_cutaway;
out vec4 frag_color;
void main() {
    if (u_cutaway && v_position.x > 0.03 && v_position.z > -0.18) {
        discard;
    }
    vec3 normal = normalize(v_normal);
    vec3 view_dir = normalize(u_camera_position - v_position);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }
    vec3 light_dir = normalize(vec3(-0.42, 0.68, 0.56));
    float diffuse = max(dot(normal, light_dir), 0.0);
    float rim = pow(max(1.0 - dot(normal, view_dir), 0.0), 2.0);
    vec3 color = v_color * (0.23 + diffuse * 0.72) + rim * vec3(0.06, 0.12, 0.15);
    frag_color = vec4(color, 1.0);
}
)GLSL";

    const uint32_t vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source);
    const uint32_t fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    if (vertex_shader == 0 || fragment_shader == 0) {
        return 0;
    }

    const uint32_t program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

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

std::vector<uint32_t> candidate_entrance_samples(const ae::PointCloud& points) {
    std::vector<uint32_t> candidates;
    candidates.reserve(points.size());
    for (uint32_t i = 0; i < points.size(); ++i) {
        if (i >= points.sample_kinds.size()) {
            continue;
        }
        const ae::PointSampleKind kind = points.sample_kinds[i];
        if (kind == ae::PointSampleKind::Center ||
            kind == ae::PointSampleKind::Spoke ||
            kind == ae::PointSampleKind::Interior) {
            candidates.push_back(i);
        }
    }
    return candidates;
}

void path_basis(ae::Vec3 direction, ae::Vec3& u, ae::Vec3& v) {
    const ae::Vec3 up = std::fabs(direction.y) < 0.88f ? ae::Vec3{0.0f, 1.0f, 0.0f} : ae::Vec3{1.0f, 0.0f, 0.0f};
    u = ae::normalize(ae::cross(up, direction));
    v = ae::normalize(ae::cross(direction, u));
}

ae::Vec3 path_sample(const MinePath& path, float t) {
    if (path.points.empty()) {
        return {};
    }
    if (path.points.size() == 1u) {
        return path.points.front();
    }
    const float scaled = saturate(t) * static_cast<float>(path.points.size() - 1u);
    const uint32_t a = std::min(static_cast<uint32_t>(scaled), static_cast<uint32_t>(path.points.size() - 2u));
    const float local_t = scaled - static_cast<float>(a);
    return ae::lerp(path.points[a], path.points[a + 1u], local_t);
}

MinePath make_wavy_path(
    ae::Vec3 start,
    ae::Vec3 end,
    float radius,
    uint32_t segments,
    uint32_t seed,
    bool primary
) {
    MinePath path;
    path.radius = radius;
    path.primary = primary;
    path.points.reserve(segments + 1u);

    const ae::Vec3 axis = ae::normalize(end - start);
    ae::Vec3 u;
    ae::Vec3 v;
    path_basis(axis, u, v);
    const float amplitude = primary ? 0.035f : 0.055f;
    const float phase_a = hash_unit(seed ^ 0x8da6b343u) * ae::Pi * 2.0f;
    const float phase_b = hash_unit(seed ^ 0xd8163841u) * ae::Pi * 2.0f;
    const float waves = primary ? 1.25f : 1.85f;

    for (uint32_t i = 0; i <= segments; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(segments);
        const float envelope = std::sin(t * ae::Pi);
        const ae::Vec3 base = ae::lerp(start, end, t);
        const ae::Vec3 wobble =
            u * (std::sin(t * waves * ae::Pi * 2.0f + phase_a) * amplitude * envelope) +
            v * (std::cos(t * (waves + 0.55f) * ae::Pi * 2.0f + phase_b) * amplitude * envelope);
        path.points.push_back(base + wobble);
    }
    return path;
}

MineNetwork build_mine_network(const ae::PointCloud& points, const MineSettings& settings) {
    MineNetwork network;
    network.chambers.push_back({{0.0f, 0.0f, 0.0f}, CoreVoidRadius, true});

    struct RankedCandidate {
        uint32_t point_index = 0;
        uint32_t rank = 0;
    };

    std::vector<RankedCandidate> ranked;
    const std::vector<uint32_t> candidates = candidate_entrance_samples(points);
    ranked.reserve(candidates.size());
    for (uint32_t point_index : candidates) {
        ranked.push_back({point_index, hash_u32(settings.seed ^ (point_index * 0x9e3779b9u))});
    }
    std::sort(ranked.begin(), ranked.end(), [](const RankedCandidate& lhs, const RankedCandidate& rhs) {
        return lhs.rank < rhs.rank;
    });

    const uint32_t entrance_count = std::min(settings.mine_density, static_cast<uint32_t>(ranked.size()));
    network.entrances.reserve(entrance_count);
    network.primary_shafts.reserve(entrance_count);
    for (uint32_t i = 0; i < entrance_count; ++i) {
        const uint32_t sample_id = ranked[i].point_index;
        const ae::Vec3 entrance_dir = ae::normalize(points.positions[sample_id]);
        network.entrances.push_back({sample_id, entrance_dir});

        ae::Vec3 u;
        ae::Vec3 v;
        path_basis(entrance_dir, u, v);
        const uint32_t shaft_seed = settings.seed ^ (i * 0x632be59bu) ^ 0x51ed270bu;
        const uint32_t segment_count = 8u + (hash_u32(shaft_seed) % 4u);
        MinePath shaft;
        shaft.primary = true;
        shaft.radius = settings.tunnel_radius * (0.86f + hash_unit(shaft_seed ^ 0x22ae35u) * 0.38f);
        shaft.points.reserve(segment_count + 1u);

        const ae::Vec3 core_bias = ae::normalize(
            entrance_dir * 0.55f +
            u * ((hash_unit(shaft_seed ^ 0xa511e9b3u) - 0.5f) * 0.55f) +
            v * ((hash_unit(shaft_seed ^ 0x63d83595u) - 0.5f) * 0.55f)
        );
        for (uint32_t step = 0; step <= segment_count; ++step) {
            const float t = static_cast<float>(step) / static_cast<float>(segment_count);
            const float radial = lerp_float(0.995f, 0.11f + hash_unit(shaft_seed ^ 0x97a1f3u) * 0.04f, std::pow(t, 0.92f));
            const float envelope = std::sin(t * ae::Pi);
            const ae::Vec3 direction = ae::normalize(
                ae::lerp(entrance_dir, core_bias, t * 0.70f) +
                u * (std::sin(t * ae::Pi * 4.0f + hash_unit(shaft_seed) * ae::Pi * 2.0f) * 0.09f * envelope) +
                v * (std::cos(t * ae::Pi * 3.3f + hash_unit(shaft_seed ^ 0x8421u) * ae::Pi * 2.0f) * 0.07f * envelope)
            );
            shaft.points.push_back(direction * radial);
        }
        network.primary_shafts.push_back(shaft);

        for (uint32_t c = 0; c < 2u; ++c) {
            const float t = 0.26f + static_cast<float>(c) * 0.31f + hash_unit(shaft_seed ^ (c * 0x6c8e9cf5u)) * 0.08f;
            network.chambers.push_back({
                path_sample(network.primary_shafts.back(), t),
                shaft.radius * (1.75f + hash_unit(shaft_seed ^ c ^ 0xc2b2ae35u) * 0.85f),
                false,
            });
        }
        network.chambers.push_back({
            path_sample(network.primary_shafts.back(), 0.80f + hash_unit(shaft_seed ^ 0x165667b1u) * 0.12f),
            shaft.radius * (1.45f + hash_unit(shaft_seed ^ 0x7feb352du) * 0.95f),
            true,
        });
    }

    const uint32_t branch_count = entrance_count * 2u;
    network.branches.reserve(branch_count);
    for (uint32_t i = 0; i < branch_count && network.primary_shafts.size() > 1u; ++i) {
        const uint32_t branch_seed = settings.seed ^ (i * 0x85ebca6bu) ^ 0xd1b54a35u;
        uint32_t a_index = hash_u32(branch_seed) % static_cast<uint32_t>(network.primary_shafts.size());
        uint32_t b_index = hash_u32(branch_seed ^ 0x27d4eb2du) % static_cast<uint32_t>(network.primary_shafts.size());
        if (a_index == b_index) {
            b_index = (b_index + 1u) % static_cast<uint32_t>(network.primary_shafts.size());
        }
        const float at = 0.18f + hash_unit(branch_seed ^ 0xa2bfe8a1u) * 0.67f;
        const float bt = 0.22f + hash_unit(branch_seed ^ 0x9e3779b9u) * 0.64f;
        const ae::Vec3 start = path_sample(network.primary_shafts[a_index], at);
        const ae::Vec3 end = path_sample(network.primary_shafts[b_index], bt);
        MinePath branch = make_wavy_path(
            start,
            end,
            settings.tunnel_radius * (0.58f + hash_unit(branch_seed ^ 0x6a09e667u) * 0.24f),
            5u + (hash_u32(branch_seed ^ 0x3c6ef372u) % 4u),
            branch_seed,
            false
        );
        network.branches.push_back(branch);

        if ((hash_u32(branch_seed ^ 0xbb67ae85u) & 3u) != 0u) {
            network.chambers.push_back({
                path_sample(network.branches.back(), 0.45f + hash_unit(branch_seed ^ 0x510e527fu) * 0.16f),
                settings.tunnel_radius * (1.15f + hash_unit(branch_seed ^ 0x1f83d9abu) * 0.85f),
                (hash_u32(branch_seed ^ 0x5be0cd19u) & 1u) != 0u,
            });
        }
    }

    return network;
}

float distance_to_segment(ae::Vec3 p, ae::Vec3 a, ae::Vec3 b) {
    const ae::Vec3 ab = b - a;
    const float len_sq = ae::dot(ab, ab);
    if (len_sq <= 0.0000001f) {
        return ae::length(p - a);
    }
    const float t = std::clamp(ae::dot(p - a, ab) / len_sq, 0.0f, 1.0f);
    return ae::length(p - (a + ab * t));
}

float mine_air_sdf(ae::Vec3 p, const MineNetwork& network) {
    float sdf = std::numeric_limits<float>::max();
    for (const MineChamber& chamber : network.chambers) {
        const float chamber_sdf = ae::length(p - chamber.center) - chamber.radius;
        sdf = smooth_min(sdf, chamber_sdf, chamber.resource ? 0.030f : 0.022f);
    }
    auto add_path = [&](const MinePath& path) {
        if (path.points.size() < 2u) {
            return;
        }
        for (uint32_t i = 0; i + 1u < path.points.size(); ++i) {
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(path.points.size() - 1u);
            const float pulse = 0.90f + 0.16f * std::sin(t * ae::Pi * 6.0f);
            const float capsule_sdf = distance_to_segment(p, path.points[i], path.points[i + 1u]) - path.radius * pulse;
            sdf = smooth_min(sdf, capsule_sdf, path.primary ? 0.018f : 0.024f);
        }
    };
    for (const MinePath& path : network.primary_shafts) {
        add_path(path);
    }
    for (const MinePath& path : network.branches) {
        add_path(path);
    }
    return sdf;
}

float solid_sdf(ae::Vec3 p, const MineNetwork& network) {
    const float sphere_sdf = ae::length(p) - PlanetRadius;
    const float air_sdf = mine_air_sdf(p, network);
    return std::max(sphere_sdf, -air_sdf);
}

ae::Vec3 material_color(ae::Vec3 p, const MineNetwork& network) {
    if (ae::length(p) > 0.935f) {
        return color_crust();
    }
    if (ae::length(p) < CoreVoidRadius + 0.055f) {
        return color_resource();
    }
    for (const MineChamber& chamber : network.chambers) {
        if (chamber.resource && ae::length(p - chamber.center) < chamber.radius + 0.045f) {
            return color_resource();
        }
    }
    return color_mine_wall();
}

ae::Vec3 estimate_normal(ae::Vec3 p, const MineNetwork& network) {
    constexpr float Epsilon = 0.0035f;
    const float dx = solid_sdf(p + ae::Vec3{Epsilon, 0.0f, 0.0f}, network) - solid_sdf(p - ae::Vec3{Epsilon, 0.0f, 0.0f}, network);
    const float dy = solid_sdf(p + ae::Vec3{0.0f, Epsilon, 0.0f}, network) - solid_sdf(p - ae::Vec3{0.0f, Epsilon, 0.0f}, network);
    const float dz = solid_sdf(p + ae::Vec3{0.0f, 0.0f, Epsilon}, network) - solid_sdf(p - ae::Vec3{0.0f, 0.0f, Epsilon}, network);
    return ae::normalize({dx, dy, dz});
}

uint32_t effective_grid_resolution(const MineSettings& settings, const MineNetwork& network) {
    (void)network;
    return std::clamp(settings.requested_grid_resolution, 64u, 512u);
}

VertexKey vertex_key(ae::Vec3 p) {
    constexpr float Scale = 50000.0f;
    return {
        static_cast<int32_t>(std::lround(p.x * Scale)),
        static_cast<int32_t>(std::lround(p.y * Scale)),
        static_cast<int32_t>(std::lround(p.z * Scale)),
    };
}

uint32_t add_vertex(
    TestbedMesh& mesh,
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash>& lookup,
    ae::Vec3 p,
    ae::Vec3 normal,
    ae::Vec3 color
) {
    const VertexKey key = vertex_key(p);
    const auto found = lookup.find(key);
    if (found != lookup.end()) {
        return found->second;
    }
    const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
    lookup.emplace(key, index);
    mesh.vertices.push_back({p, normal, color});
    return index;
}

ae::Vec3 interpolate_edge(ae::Vec3 a, ae::Vec3 b, float va, float vb) {
    const float denom = va - vb;
    if (std::fabs(denom) <= 0.0000001f) {
        return (a + b) * 0.5f;
    }
    return ae::lerp(a, b, std::clamp(va / denom, 0.0f, 1.0f));
}

void append_triangle(TestbedMesh& mesh, uint32_t a, uint32_t b, uint32_t c) {
    if (a == b || b == c || c == a) {
        ++mesh.stats.degenerate_triangles;
        return;
    }
    const ae::Vec3 pa = mesh.vertices[a].position;
    const ae::Vec3 pb = mesh.vertices[b].position;
    const ae::Vec3 pc = mesh.vertices[c].position;
    const ae::Vec3 face = ae::cross(pb - pa, pc - pa);
    if (ae::length(face) <= 0.0000008f) {
        ++mesh.stats.degenerate_triangles;
        return;
    }
    const ae::Vec3 average_normal = ae::normalize(mesh.vertices[a].normal + mesh.vertices[b].normal + mesh.vertices[c].normal);
    if (ae::dot(face, average_normal) < 0.0f) {
        std::swap(b, c);
    }
    mesh.indices.push_back(a);
    mesh.indices.push_back(b);
    mesh.indices.push_back(c);
}

BuildStats validate_mesh(const TestbedMesh& mesh, const MineSettings& settings, const MineNetwork& network, uint32_t resolution) {
    BuildStats stats;
    stats.seed = settings.seed;
    stats.entrances = static_cast<uint32_t>(network.entrances.size());
    stats.primary_shafts = static_cast<uint32_t>(network.primary_shafts.size());
    stats.branch_tunnels = static_cast<uint32_t>(network.branches.size());
    stats.chambers = static_cast<uint32_t>(network.chambers.size());
    stats.grid_resolution = resolution;
    stats.vertices = static_cast<uint32_t>(mesh.vertices.size());
    stats.triangles = static_cast<uint32_t>(mesh.indices.size() / 3u);

    for (const RenderVertex& vertex : mesh.vertices) {
        if (!finite_vec3(vertex.position) || !finite_vec3(vertex.normal)) {
            ++stats.invalid_indices;
        }
    }

    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edge_counts;
    edge_counts.reserve(mesh.indices.size());
    for (uint32_t i = 0; i + 2u < mesh.indices.size(); i += 3u) {
        const uint32_t a = mesh.indices[i];
        const uint32_t b = mesh.indices[i + 1u];
        const uint32_t c = mesh.indices[i + 2u];
        if (a >= mesh.vertices.size() || b >= mesh.vertices.size() || c >= mesh.vertices.size()) {
            ++stats.invalid_indices;
            continue;
        }
        if (a == b || b == c || c == a ||
            ae::length(ae::cross(mesh.vertices[b].position - mesh.vertices[a].position, mesh.vertices[c].position - mesh.vertices[a].position)) <= 0.0000008f) {
            ++stats.degenerate_triangles;
            continue;
        }
        ++edge_counts[make_edge_key(a, b)];
        ++edge_counts[make_edge_key(b, c)];
        ++edge_counts[make_edge_key(c, a)];
    }
    for (const auto& entry : edge_counts) {
        if (entry.second == 1u) {
            ++stats.open_edges;
        } else if (entry.second != 2u) {
            ++stats.nonmanifold_edges;
        }
    }

    stats.ok = stats.invalid_indices == 0u && stats.degenerate_triangles == 0u;
    std::ostringstream message;
    message << (stats.ok ? "Mesh validation OK" : "Mesh validation FAILED")
            << ": seed " << stats.seed
            << ", entrances " << stats.entrances
            << ", primary shafts " << stats.primary_shafts
            << ", branches " << stats.branch_tunnels
            << ", chambers " << stats.chambers
            << ", grid " << stats.grid_resolution
            << ", vertices " << stats.vertices
            << ", triangles " << stats.triangles
            << ", invalid " << stats.invalid_indices
            << ", degenerate " << stats.degenerate_triangles
            << ", open " << stats.open_edges
            << ", nonmanifold " << stats.nonmanifold_edges;
    stats.message = message.str();
    return stats;
}

uint32_t part1by2(uint32_t value) {
    value &= 0x000003ffu;
    value = (value ^ (value << 16u)) & 0xff0000ffu;
    value = (value ^ (value << 8u)) & 0x0300f00fu;
    value = (value ^ (value << 4u)) & 0x030c30c3u;
    value = (value ^ (value << 2u)) & 0x09249249u;
    return value;
}

uint64_t morton_encode_chunk(uint32_t x, uint32_t y, uint32_t z) {
    return static_cast<uint64_t>(part1by2(x)) |
           (static_cast<uint64_t>(part1by2(y)) << 1u) |
           (static_cast<uint64_t>(part1by2(z)) << 2u);
}

uint32_t lod_resolution(uint32_t lod) {
    return 128u << std::min(lod, 2u);
}

uint32_t lod_chunks_per_axis(uint32_t lod, uint32_t chunk_size) {
    return (lod_resolution(lod) + chunk_size - 1u) / chunk_size;
}

MortonChunkKey make_chunk_key(uint32_t lod, uint32_t x, uint32_t y, uint32_t z) {
    const uint64_t lod_bits = static_cast<uint64_t>(std::min(lod, 2u)) << 60u;
    return {lod_bits | morton_encode_chunk(x, y, z), std::min(lod, 2u), x, y, z};
}

uint32_t chunk_sample_resolution(uint32_t chunk_size) {
    return chunk_size + 1u;
}

size_t chunk_sample_index(uint32_t chunk_size, uint32_t x, uint32_t y, uint32_t z) {
    const uint32_t sample_resolution = chunk_sample_resolution(chunk_size);
    return static_cast<size_t>(z) * sample_resolution * sample_resolution +
           static_cast<size_t>(y) * sample_resolution +
           static_cast<size_t>(x);
}

ae::Vec3 field_position(uint32_t resolution, uint32_t x, uint32_t y, uint32_t z) {
    const float voxel = (GridRadius * 2.0f) / static_cast<float>(resolution);
    return {
        -GridRadius + static_cast<float>(x) * voxel,
        -GridRadius + static_cast<float>(y) * voxel,
        -GridRadius + static_cast<float>(z) * voxel,
    };
}

ae::Vec3 field_position(const MineSdfField& field, uint32_t x, uint32_t y, uint32_t z) {
    return field_position(field.resolution, x, y, z);
}

int32_t field_coord(const MineSdfField& field, float value) {
    return static_cast<int32_t>(std::floor((value + GridRadius) / field.voxel));
}

uint32_t clamp_point(const MineSdfField& field, int32_t value) {
    return static_cast<uint32_t>(std::clamp(value, 0, static_cast<int32_t>(field.point_resolution - 1u)));
}

void chunk_world_bounds(uint32_t resolution, uint32_t chunk_size, MortonChunkKey key, ae::Vec3& minimum, ae::Vec3& maximum) {
    const uint32_t min_x = key.x * chunk_size;
    const uint32_t min_y = key.y * chunk_size;
    const uint32_t min_z = key.z * chunk_size;
    const uint32_t max_x = std::min(resolution, (key.x + 1u) * chunk_size);
    const uint32_t max_y = std::min(resolution, (key.y + 1u) * chunk_size);
    const uint32_t max_z = std::min(resolution, (key.z + 1u) * chunk_size);
    minimum = field_position(resolution, min_x, min_y, min_z);
    maximum = field_position(resolution, max_x, max_y, max_z);
}

void chunk_world_bounds(uint32_t chunk_size, MortonChunkKey key, ae::Vec3& minimum, ae::Vec3& maximum) {
    chunk_world_bounds(lod_resolution(key.lod), chunk_size, key, minimum, maximum);
}

bool boxes_overlap(ae::Vec3 a_min, ae::Vec3 a_max, ae::Vec3 b_min, ae::Vec3 b_max) {
    return a_min.x <= b_max.x && a_max.x >= b_min.x &&
           a_min.y <= b_max.y && a_max.y >= b_min.y &&
           a_min.z <= b_max.z && a_max.z >= b_min.z;
}

float min_distance_to_box_origin(ae::Vec3 minimum, ae::Vec3 maximum) {
    auto axis_distance = [](float min_value, float max_value) {
        if (0.0f < min_value) return min_value;
        if (0.0f > max_value) return -max_value;
        return 0.0f;
    };
    const ae::Vec3 closest{
        axis_distance(minimum.x, maximum.x),
        axis_distance(minimum.y, maximum.y),
        axis_distance(minimum.z, maximum.z),
    };
    return ae::length(closest);
}

float max_distance_to_box_origin(ae::Vec3 minimum, ae::Vec3 maximum) {
    float max_dist_sq = 0.0f;
    for (uint32_t z = 0; z < 2u; ++z) {
        for (uint32_t y = 0; y < 2u; ++y) {
            for (uint32_t x = 0; x < 2u; ++x) {
                const ae::Vec3 p{
                    x == 0u ? minimum.x : maximum.x,
                    y == 0u ? minimum.y : maximum.y,
                    z == 0u ? minimum.z : maximum.z,
                };
                max_dist_sq = std::max(max_dist_sq, ae::dot(p, p));
            }
        }
    }
    return std::sqrt(max_dist_sq);
}

float primitive_sdf(ae::Vec3 p, const SdfPrimitive& primitive) {
    if (primitive.shape == SdfEditShape::Capsule) {
        return distance_to_segment(p, primitive.a, primitive.b) - primitive.radius;
    }
    return ae::length(p - primitive.center) - primitive.radius;
}

SdfPrimitive primitive_from_edit(const SdfEdit& edit) {
    SdfPrimitive primitive;
    primitive.mode = edit.mode;
    primitive.shape = edit.shape;
    primitive.center = edit.center;
    primitive.a = edit.a;
    primitive.b = edit.b;
    primitive.radius = edit.radius;
    primitive.blend = edit.blend;
    primitive.material = edit.material;
    const float extent = edit.radius + edit.blend + 0.018f;
    if (edit.shape == SdfEditShape::Capsule) {
        primitive.minimum = {
            std::min(edit.a.x, edit.b.x) - extent,
            std::min(edit.a.y, edit.b.y) - extent,
            std::min(edit.a.z, edit.b.z) - extent,
        };
        primitive.maximum = {
            std::max(edit.a.x, edit.b.x) + extent,
            std::max(edit.a.y, edit.b.y) + extent,
            std::max(edit.a.z, edit.b.z) + extent,
        };
    } else {
        primitive.minimum = edit.center - ae::Vec3{extent, extent, extent};
        primitive.maximum = edit.center + ae::Vec3{extent, extent, extent};
    }
    return primitive;
}

SdfPrimitive sphere_primitive(ae::Vec3 center, float radius, float blend, EditMaterial material) {
    SdfEdit edit;
    edit.mode = SdfEditMode::Carve;
    edit.shape = SdfEditShape::Sphere;
    edit.center = center;
    edit.a = center;
    edit.b = center;
    edit.radius = radius;
    edit.blend = blend;
    edit.material = material;
    return primitive_from_edit(edit);
}

SdfPrimitive capsule_primitive(ae::Vec3 a, ae::Vec3 b, float radius, float blend, EditMaterial material) {
    SdfEdit edit;
    edit.mode = SdfEditMode::Carve;
    edit.shape = SdfEditShape::Capsule;
    edit.center = (a + b) * 0.5f;
    edit.a = a;
    edit.b = b;
    edit.radius = radius;
    edit.blend = blend;
    edit.material = material;
    return primitive_from_edit(edit);
}

float evaluate_sparse_sdf(ae::Vec3 p, const std::vector<SdfPrimitive>& procedural, const std::vector<SdfPrimitive>& edits) {
    float value = ae::length(p) - PlanetRadius;
    auto apply = [&](const SdfPrimitive& primitive) {
        const float sdf = primitive_sdf(p, primitive);
        if (sdf > primitive.blend) {
            return;
        }
        if (primitive.mode == SdfEditMode::Carve) {
            value = std::max(value, -sdf);
        } else {
            value = std::min(value, sdf);
        }
    };
    for (const SdfPrimitive& primitive : procedural) {
        apply(primitive);
    }
    for (const SdfPrimitive& primitive : edits) {
        apply(primitive);
    }
    return value;
}

EditMaterial evaluate_sparse_material(ae::Vec3 p, const std::vector<SdfPrimitive>& procedural, const std::vector<SdfPrimitive>& edits) {
    EditMaterial material = ae::length(p) > 0.935f ? EditMaterial::Crust : EditMaterial::MineWall;
    for (const SdfPrimitive& primitive : procedural) {
        if (primitive_sdf(p, primitive) <= 0.0f) {
            material = primitive.material;
        }
    }
    for (const SdfPrimitive& primitive : edits) {
        if (primitive_sdf(p, primitive) <= 0.0f) {
            material = primitive.mode == SdfEditMode::Fill ? EditMaterial::Fill : primitive.material;
        }
    }
    if (ae::length(p) < CoreVoidRadius + 0.055f) {
        material = EditMaterial::Resource;
    }
    return material;
}

float sample_field(const MineSdfField& field, ae::Vec3 p) {
    return evaluate_sparse_sdf(p, field.procedural_primitives, field.edit_primitives);
}

ae::Vec3 field_normal(const MineSdfField& field, ae::Vec3 p) {
    const float h = field.voxel;
    return ae::normalize({
        sample_field(field, p + ae::Vec3{h, 0.0f, 0.0f}) - sample_field(field, p - ae::Vec3{h, 0.0f, 0.0f}),
        sample_field(field, p + ae::Vec3{0.0f, h, 0.0f}) - sample_field(field, p - ae::Vec3{0.0f, h, 0.0f}),
        sample_field(field, p + ae::Vec3{0.0f, 0.0f, h}) - sample_field(field, p - ae::Vec3{0.0f, 0.0f, h}),
    });
}

ae::Vec3 material_color(EditMaterial tag, ae::Vec3 p) {
    if (tag == EditMaterial::Fill) {
        return {0.16f, 0.34f, 0.22f};
    }
    if (tag == EditMaterial::Resource) {
        return color_resource();
    }
    if (tag == EditMaterial::Crust || ae::length(p) > 0.935f) {
        return color_crust();
    }
    return color_mine_wall();
}

std::vector<uint64_t> mark_dirty_box(MineSdfField& field, ae::Vec3 minimum, ae::Vec3 maximum, bool urgent = false) {
    std::vector<uint64_t> touched;
    for (auto& entry : field.chunks) {
        ae::Vec3 chunk_min;
        ae::Vec3 chunk_max;
        chunk_world_bounds(field.chunk_size, entry.second.key, chunk_min, chunk_max);
        if (boxes_overlap(minimum, maximum, chunk_min, chunk_max)) {
            entry.second.dirty = true;
            entry.second.queued = false;
            entry.second.gpu_dirty = true;
            entry.second.urgent = entry.second.urgent || urgent;
            ++entry.second.build_serial;
            entry.second.state = ChunkState::Empty;
            touched.push_back(entry.first);
        }
    }
    return touched;
}

void mark_all_chunks_dirty(MineSdfField& field) {
    for (auto& entry : field.chunks) {
        entry.second.dirty = true;
        entry.second.queued = false;
        entry.second.gpu_dirty = true;
        entry.second.urgent = false;
        ++entry.second.build_serial;
        entry.second.state = ChunkState::Empty;
    }
}

uint32_t dirty_chunk_count(const MineSdfField& field) {
    uint32_t count = 0;
    for (const auto& entry : field.chunks) {
        count += entry.second.dirty ? 1u : 0u;
    }
    return count;
}

void add_active_chunk(MineSdfField& field, MortonChunkKey key) {
    if (field.chunks.find(key.code) != field.chunks.end()) {
        return;
    }
    const uint32_t resolution = lod_resolution(key.lod);
    MineSdfChunk chunk;
    chunk.key = key;
    chunk.min_x = key.x * field.chunk_size;
    chunk.min_y = key.y * field.chunk_size;
    chunk.min_z = key.z * field.chunk_size;
    chunk.max_x = std::min(resolution, (key.x + 1u) * field.chunk_size);
    chunk.max_y = std::min(resolution, (key.y + 1u) * field.chunk_size);
    chunk.max_z = std::min(resolution, (key.z + 1u) * field.chunk_size);
    field.active_keys.push_back(key);
    field.chunks.emplace(key.code, std::move(chunk));
}

int32_t lod_coord(uint32_t lod, float value) {
    const float voxel = (GridRadius * 2.0f) / static_cast<float>(lod_resolution(lod));
    return static_cast<int32_t>(std::floor((value + GridRadius) / voxel));
}

void add_active_chunks_for_box(MineSdfField& field, uint32_t lod, ae::Vec3 minimum, ae::Vec3 maximum) {
    const uint32_t chunks_per_axis = lod_chunks_per_axis(lod, field.chunk_size);
    const uint32_t min_x = static_cast<uint32_t>(std::clamp(lod_coord(lod, minimum.x) / static_cast<int32_t>(field.chunk_size), 0, static_cast<int32_t>(chunks_per_axis - 1u)));
    const uint32_t min_y = static_cast<uint32_t>(std::clamp(lod_coord(lod, minimum.y) / static_cast<int32_t>(field.chunk_size), 0, static_cast<int32_t>(chunks_per_axis - 1u)));
    const uint32_t min_z = static_cast<uint32_t>(std::clamp(lod_coord(lod, minimum.z) / static_cast<int32_t>(field.chunk_size), 0, static_cast<int32_t>(chunks_per_axis - 1u)));
    const uint32_t max_x = static_cast<uint32_t>(std::clamp((lod_coord(lod, maximum.x) + static_cast<int32_t>(field.chunk_size)) / static_cast<int32_t>(field.chunk_size), 0, static_cast<int32_t>(chunks_per_axis - 1u)));
    const uint32_t max_y = static_cast<uint32_t>(std::clamp((lod_coord(lod, maximum.y) + static_cast<int32_t>(field.chunk_size)) / static_cast<int32_t>(field.chunk_size), 0, static_cast<int32_t>(chunks_per_axis - 1u)));
    const uint32_t max_z = static_cast<uint32_t>(std::clamp((lod_coord(lod, maximum.z) + static_cast<int32_t>(field.chunk_size)) / static_cast<int32_t>(field.chunk_size), 0, static_cast<int32_t>(chunks_per_axis - 1u)));
    for (uint32_t z = min_z; z <= max_z; ++z) {
        for (uint32_t y = min_y; y <= max_y; ++y) {
            for (uint32_t x = min_x; x <= max_x; ++x) {
                add_active_chunk(field, make_chunk_key(lod, x, y, z));
            }
        }
    }
}

struct SdfLodPolicy {
    float near_radius = 0.34f;
    float mid_radius = 0.78f;
    float far_radius = 1.35f;
    float target_fps = 60.0f;
    float upload_budget_ms = 2.0f;
};

void add_focus_lod_chunks(MineSdfField& field, ae::Vec3 focus, const SdfLodPolicy& policy) {
    const ae::Vec3 mid_extent{policy.mid_radius, policy.mid_radius, policy.mid_radius};
    const ae::Vec3 near_extent{policy.near_radius, policy.near_radius, policy.near_radius};
    add_active_chunks_for_box(field, 1u, focus - mid_extent, focus + mid_extent);
    add_active_chunks_for_box(field, 2u, focus - near_extent, focus + near_extent);
}

void build_procedural_primitives(MineSdfField& field, const MineNetwork& network) {
    field.procedural_primitives.clear();
    for (const MineChamber& chamber : network.chambers) {
        field.procedural_primitives.push_back(sphere_primitive(
            chamber.center,
            chamber.radius,
            chamber.resource ? 0.030f : 0.022f,
            chamber.resource ? EditMaterial::Resource : EditMaterial::MineWall
        ));
    }
    auto add_path = [&](const MinePath& path) {
        if (path.points.size() < 2u) {
            return;
        }
        for (uint32_t i = 0; i + 1u < path.points.size(); ++i) {
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(path.points.size() - 1u);
            const float radius = path.radius * (0.90f + 0.16f * std::sin(t * ae::Pi * 6.0f));
            field.procedural_primitives.push_back(capsule_primitive(
                path.points[i],
                path.points[i + 1u],
                radius,
                path.primary ? 0.018f : 0.024f,
                EditMaterial::MineWall
            ));
        }
    };
    for (const MinePath& path : network.primary_shafts) {
        add_path(path);
    }
    for (const MinePath& path : network.branches) {
        add_path(path);
    }
}

MineSdfField build_base_mine_field(const MineSettings& settings, const MineNetwork& network) {
    MineSdfField field;
    field.resolution = effective_grid_resolution(settings, network);
    field.point_resolution = field.resolution + 1u;
    field.voxel = (GridRadius * 2.0f) / static_cast<float>(field.resolution);
    field.brush_radius = settings.tunnel_radius;
    field.chunks_x = (field.resolution + field.chunk_size - 1u) / field.chunk_size;
    field.chunks_y = field.chunks_x;
    field.chunks_z = field.chunks_x;
    build_procedural_primitives(field, network);

    for (const SdfPrimitive& primitive : field.procedural_primitives) {
        add_active_chunks_for_box(field, 0u, primitive.minimum, primitive.maximum);
        add_active_chunks_for_box(field, 1u, primitive.minimum, primitive.maximum);
    }
    const uint32_t coarse_chunks = lod_chunks_per_axis(0u, field.chunk_size);
    for (uint32_t z = 0; z < coarse_chunks; ++z) {
        for (uint32_t y = 0; y < coarse_chunks; ++y) {
            for (uint32_t x = 0; x < coarse_chunks; ++x) {
                ae::Vec3 minimum;
                ae::Vec3 maximum;
                const MortonChunkKey key = make_chunk_key(0u, x, y, z);
                chunk_world_bounds(field.chunk_size, key, minimum, maximum);
                if (min_distance_to_box_origin(minimum, maximum) <= PlanetRadius &&
                    max_distance_to_box_origin(minimum, maximum) >= PlanetRadius) {
                    add_active_chunk(field, key);
                }
            }
        }
    }
    std::sort(field.active_keys.begin(), field.active_keys.end(), [](const MortonChunkKey& lhs, const MortonChunkKey& rhs) {
        return lhs.code < rhs.code;
    });
    return field;
}

void rebuild_field_from_history(MineSdfField& field) {
    field.edit_primitives.clear();
    for (EditableMineObject& object : field.objects) {
        object.enabled = false;
    }
    for (const SdfEdit& edit : field.edits) {
        const SdfPrimitive primitive = primitive_from_edit(edit);
        field.edit_primitives.push_back(primitive);
        add_active_chunks_for_box(field, 0u, primitive.minimum, primitive.maximum);
        add_active_chunks_for_box(field, 1u, primitive.minimum, primitive.maximum);
        add_active_chunks_for_box(field, 2u, primitive.minimum, primitive.maximum);
        if (edit.mode == SdfEditMode::Carve && edit.object_id != 0u) {
            for (EditableMineObject& object : field.objects) {
                if (object.id == edit.object_id) {
                    object.enabled = true;
                }
            }
        }
        if (edit.mode == SdfEditMode::Fill && edit.object_id != 0u) {
            for (EditableMineObject& object : field.objects) {
                if (object.id == edit.object_id) {
                    object.enabled = false;
                }
            }
        }
    }
    mark_all_chunks_dirty(field);
}

SdfEdit make_sphere_edit(MineSdfField& field, SdfEditMode mode, ae::Vec3 center, float radius, EditMaterial material, uint32_t object_id = 0u) {
    return {
        field.next_edit_id++,
        object_id,
        mode,
        SdfEditShape::Sphere,
        center,
        center,
        center,
        radius,
        std::max(field.voxel * 1.5f, radius * 0.30f),
        material,
    };
}

SdfEdit make_capsule_edit(MineSdfField& field, SdfEditMode mode, ae::Vec3 a, ae::Vec3 b, float radius, EditMaterial material, uint32_t object_id = 0u) {
    return {
        field.next_edit_id++,
        object_id,
        mode,
        SdfEditShape::Capsule,
        (a + b) * 0.5f,
        a,
        b,
        radius,
        std::max(field.voxel * 1.5f, radius * 0.30f),
        material,
    };
}

std::vector<uint64_t> apply_sdf_edit(MineSdfField& field, const SdfEdit& edit) {
    field.edits.push_back(edit);
    field.redo_edits.clear();
    const SdfPrimitive primitive = primitive_from_edit(edit);
    field.edit_primitives.push_back(primitive);
    add_active_chunks_for_box(field, 0u, primitive.minimum, primitive.maximum);
    add_active_chunks_for_box(field, 1u, primitive.minimum, primitive.maximum);
    add_active_chunks_for_box(field, 2u, primitive.minimum, primitive.maximum);
    std::vector<uint64_t> touched = mark_dirty_box(field, primitive.minimum, primitive.maximum, true);
    if (edit.mode == SdfEditMode::Fill && edit.object_id != 0u) {
        for (EditableMineObject& object : field.objects) {
            if (object.id == edit.object_id) {
                object.enabled = false;
            }
        }
    }
    return touched;
}

bool undo_edit(MineSdfField& field) {
    if (field.edits.empty()) {
        return false;
    }
    field.redo_edits.push_back(field.edits.back());
    field.edits.pop_back();
    rebuild_field_from_history(field);
    return true;
}

bool redo_edit(MineSdfField& field) {
    if (field.redo_edits.empty()) {
        return false;
    }
    const SdfEdit edit = field.redo_edits.back();
    field.redo_edits.pop_back();
    field.edits.push_back(edit);
    rebuild_field_from_history(field);
    return true;
}

struct ChunkBuildContext {
    uint32_t generation = 0;
    uint32_t resolution = 512;
    uint32_t chunk_size = 16;
    std::vector<SdfPrimitive> procedural_primitives;
    std::vector<SdfPrimitive> edit_primitives;
};

float sample_context_field(const ChunkBuildContext& context, ae::Vec3 p) {
    return evaluate_sparse_sdf(p, context.procedural_primitives, context.edit_primitives);
}

ae::Vec3 context_normal(const ChunkBuildContext& context, ae::Vec3 p) {
    const float h = (GridRadius * 2.0f) / static_cast<float>(context.resolution);
    return ae::normalize({
        sample_context_field(context, p + ae::Vec3{h, 0.0f, 0.0f}) - sample_context_field(context, p - ae::Vec3{h, 0.0f, 0.0f}),
        sample_context_field(context, p + ae::Vec3{0.0f, h, 0.0f}) - sample_context_field(context, p - ae::Vec3{0.0f, h, 0.0f}),
        sample_context_field(context, p + ae::Vec3{0.0f, 0.0f, h}) - sample_context_field(context, p - ae::Vec3{0.0f, 0.0f, h}),
    });
}

ChunkBuildResult build_chunk_result(const ChunkBuildContext& context, MortonChunkKey key, uint32_t build_serial) {
    static constexpr std::array<std::array<uint32_t, 3u>, 8u> CornerOffset = {{
        {{0u, 0u, 0u}}, {{1u, 0u, 0u}}, {{1u, 1u, 0u}}, {{0u, 1u, 0u}},
        {{0u, 0u, 1u}}, {{1u, 0u, 1u}}, {{1u, 1u, 1u}}, {{0u, 1u, 1u}},
    }};
    static constexpr std::array<std::array<uint32_t, 2u>, 12u> EdgeCorners = {{
        {{0u, 1u}}, {{1u, 2u}}, {{2u, 3u}}, {{3u, 0u}},
        {{4u, 5u}}, {{5u, 6u}}, {{6u, 7u}}, {{7u, 4u}},
        {{0u, 4u}}, {{1u, 5u}}, {{2u, 6u}}, {{3u, 7u}},
    }};

    ChunkBuildResult result;
    result.generation = context.generation;
    result.build_serial = build_serial;
    result.key = key;
    result.chunk.key = key;
    result.chunk.build_serial = build_serial;
    const uint32_t chunk_resolution = lod_resolution(key.lod);
    result.chunk.min_x = key.x * context.chunk_size;
    result.chunk.min_y = key.y * context.chunk_size;
    result.chunk.min_z = key.z * context.chunk_size;
    result.chunk.max_x = std::min(chunk_resolution, (key.x + 1u) * context.chunk_size);
    result.chunk.max_y = std::min(chunk_resolution, (key.y + 1u) * context.chunk_size);
    result.chunk.max_z = std::min(chunk_resolution, (key.z + 1u) * context.chunk_size);
    result.chunk.state = ChunkState::Ready;
    result.chunk.dirty = false;
    result.chunk.queued = false;

    ae::Vec3 chunk_min;
    ae::Vec3 chunk_max;
    chunk_world_bounds(context.chunk_size, key, chunk_min, chunk_max);
    std::vector<SdfPrimitive> relevant_procedural;
    std::vector<SdfPrimitive> relevant_edits;
    relevant_procedural.reserve(context.procedural_primitives.size());
    relevant_edits.reserve(context.edit_primitives.size());
    for (const SdfPrimitive& primitive : context.procedural_primitives) {
        if (boxes_overlap(chunk_min, chunk_max, primitive.minimum, primitive.maximum)) {
            relevant_procedural.push_back(primitive);
        }
    }
    for (const SdfPrimitive& primitive : context.edit_primitives) {
        if (boxes_overlap(chunk_min, chunk_max, primitive.minimum, primitive.maximum)) {
            relevant_edits.push_back(primitive);
        }
    }
    auto relevant_normal = [&](ae::Vec3 p) {
        const float h = (GridRadius * 2.0f) / static_cast<float>(chunk_resolution);
        return ae::normalize({
            evaluate_sparse_sdf(p + ae::Vec3{h, 0.0f, 0.0f}, relevant_procedural, relevant_edits) -
                evaluate_sparse_sdf(p - ae::Vec3{h, 0.0f, 0.0f}, relevant_procedural, relevant_edits),
            evaluate_sparse_sdf(p + ae::Vec3{0.0f, h, 0.0f}, relevant_procedural, relevant_edits) -
                evaluate_sparse_sdf(p - ae::Vec3{0.0f, h, 0.0f}, relevant_procedural, relevant_edits),
            evaluate_sparse_sdf(p + ae::Vec3{0.0f, 0.0f, h}, relevant_procedural, relevant_edits) -
                evaluate_sparse_sdf(p - ae::Vec3{0.0f, 0.0f, h}, relevant_procedural, relevant_edits),
        });
    };

    const uint32_t sample_resolution = chunk_sample_resolution(context.chunk_size);
    result.chunk.values.resize(static_cast<size_t>(sample_resolution) * sample_resolution * sample_resolution);
    result.chunk.material_tags.resize(result.chunk.values.size(), material_tag(EditMaterial::Auto));
    for (uint32_t z = 0; z < sample_resolution; ++z) {
        for (uint32_t y = 0; y < sample_resolution; ++y) {
            for (uint32_t x = 0; x < sample_resolution; ++x) {
                const uint32_t gx = result.chunk.min_x + x;
                const uint32_t gy = result.chunk.min_y + y;
                const uint32_t gz = result.chunk.min_z + z;
                const ae::Vec3 p = field_position(chunk_resolution, gx, gy, gz);
                const size_t sample_index = chunk_sample_index(context.chunk_size, x, y, z);
                result.chunk.values[sample_index] = evaluate_sparse_sdf(p, relevant_procedural, relevant_edits);
                result.chunk.material_tags[sample_index] = material_tag(evaluate_sparse_material(p, relevant_procedural, relevant_edits));
            }
        }
    }

    auto chunk_value = [&](uint32_t x, uint32_t y, uint32_t z) {
        return result.chunk.values[chunk_sample_index(context.chunk_size, x, y, z)];
    };
    auto chunk_material = [&](uint32_t x, uint32_t y, uint32_t z) {
        return tag_material(result.chunk.material_tags[chunk_sample_index(context.chunk_size, x, y, z)]);
    };

    TestbedMesh local;
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> vertex_lookup;
    vertex_lookup.reserve((context.chunk_size + 1u) * (context.chunk_size + 1u) * 4u);
    for (uint32_t z = 0; z < result.chunk.max_z - result.chunk.min_z; ++z) {
        for (uint32_t y = 0; y < result.chunk.max_y - result.chunk.min_y; ++y) {
            for (uint32_t x = 0; x < result.chunk.max_x - result.chunk.min_x; ++x) {
                std::array<float, 8u> cube_values = {};
                std::array<ae::Vec3, 8u> cube_positions = {};
                uint32_t cube_index_value = 0;
                for (uint32_t corner = 0; corner < 8u; ++corner) {
                    const uint32_t cx = x + CornerOffset[corner][0u];
                    const uint32_t cy = y + CornerOffset[corner][1u];
                    const uint32_t cz = z + CornerOffset[corner][2u];
                    cube_values[corner] = chunk_value(cx, cy, cz);
                    cube_positions[corner] = field_position(
                        chunk_resolution,
                        result.chunk.min_x + cx,
                        result.chunk.min_y + cy,
                        result.chunk.min_z + cz
                    );
                    if (cube_values[corner] < 0.0f) {
                        cube_index_value |= 1u << corner;
                    }
                }
                const int edge_mask = ae::mc_tables::EdgeTable[cube_index_value];
                if (edge_mask == 0) {
                    continue;
                }
                std::array<uint32_t, 12u> edge_vertices = {};
                for (uint32_t edge = 0; edge < 12u; ++edge) {
                    if ((edge_mask & (1 << edge)) == 0) {
                        continue;
                    }
                    const uint32_t a = EdgeCorners[edge][0u];
                    const uint32_t b = EdgeCorners[edge][1u];
                    const ae::Vec3 p = interpolate_edge(cube_positions[a], cube_positions[b], cube_values[a], cube_values[b]);
                    const EditMaterial tag = chunk_material(
                        std::min<uint32_t>(context.chunk_size, x + CornerOffset[a][0u]),
                        std::min<uint32_t>(context.chunk_size, y + CornerOffset[a][1u]),
                        std::min<uint32_t>(context.chunk_size, z + CornerOffset[a][2u])
                    );
                    edge_vertices[edge] = add_vertex(local, vertex_lookup, p, relevant_normal(p), material_color(tag, p));
                }
                for (uint32_t tri = 0; tri < 16u && ae::mc_tables::TriTable[cube_index_value][tri] != -1; tri += 3u) {
                    append_triangle(
                        local,
                        edge_vertices[static_cast<uint32_t>(ae::mc_tables::TriTable[cube_index_value][tri])],
                        edge_vertices[static_cast<uint32_t>(ae::mc_tables::TriTable[cube_index_value][tri + 1u])],
                        edge_vertices[static_cast<uint32_t>(ae::mc_tables::TriTable[cube_index_value][tri + 2u])]
                    );
                }
            }
        }
    }
    result.chunk.vertices = std::move(local.vertices);
    result.chunk.indices = std::move(local.indices);
    result.chunk.invalid_indices = 0;
    result.chunk.degenerate_triangles = 0;
    return result;
}

bool chunk_ready(const MineSdfField& field, MortonChunkKey key) {
    const auto found = field.chunks.find(key.code);
    return found != field.chunks.end() && found->second.state == ChunkState::Ready;
}

MortonChunkKey parent_chunk_key(MortonChunkKey key) {
    return make_chunk_key(key.lod - 1u, key.x / 2u, key.y / 2u, key.z / 2u);
}

bool all_child_chunks_ready(const MineSdfField& field, MortonChunkKey parent) {
    if (parent.lod >= 2u) {
        return false;
    }
    const uint32_t child_lod = parent.lod + 1u;
    const uint32_t child_base_x = parent.x * 2u;
    const uint32_t child_base_y = parent.y * 2u;
    const uint32_t child_base_z = parent.z * 2u;
    for (uint32_t z = 0; z < 2u; ++z) {
        for (uint32_t y = 0; y < 2u; ++y) {
            for (uint32_t x = 0; x < 2u; ++x) {
                if (!chunk_ready(field, make_chunk_key(child_lod, child_base_x + x, child_base_y + y, child_base_z + z))) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool should_draw_lod_chunk(const MineSdfField& field, MortonChunkKey key) {
    if (!chunk_ready(field, key)) {
        return false;
    }
    if (key.lod > 0u) {
        const MortonChunkKey parent = parent_chunk_key(key);
        if (chunk_ready(field, parent) && !all_child_chunks_ready(field, parent)) {
            return false;
        }
    }
    if (all_child_chunks_ready(field, key)) {
        return false;
    }
    return true;
}

std::vector<MortonChunkKey> collect_visible_lod_keys(MineSdfField& field) {
    field.visible_lod_counts = {0u, 0u, 0u};
    std::vector<MortonChunkKey> visible_keys;
    visible_keys.reserve(field.active_keys.size());
    for (const MortonChunkKey& key : field.active_keys) {
        if (!should_draw_lod_chunk(field, key)) {
            continue;
        }
        visible_keys.push_back(key);
        if (key.lod < field.visible_lod_counts.size()) {
            ++field.visible_lod_counts[key.lod];
        }
    }
    return visible_keys;
}

bool chunk_gpu_ready(const MineSdfField& field, const GpuChunkStore& store, MortonChunkKey key) {
    const auto chunk = field.chunks.find(key.code);
    if (chunk == field.chunks.end() || chunk->second.state != ChunkState::Ready || chunk->second.gpu_dirty) {
        return false;
    }
    const auto gpu_chunk = store.chunks.find(key.code);
    return gpu_chunk != store.chunks.end() && gpu_chunk->second.generation == field.generation && gpu_chunk->second.index_count > 0u;
}

bool all_child_chunks_gpu_ready(const MineSdfField& field, const GpuChunkStore& store, MortonChunkKey parent) {
    if (parent.lod >= 2u) {
        return false;
    }
    const uint32_t child_lod = parent.lod + 1u;
    const uint32_t child_base_x = parent.x * 2u;
    const uint32_t child_base_y = parent.y * 2u;
    const uint32_t child_base_z = parent.z * 2u;
    for (uint32_t z = 0; z < 2u; ++z) {
        for (uint32_t y = 0; y < 2u; ++y) {
            for (uint32_t x = 0; x < 2u; ++x) {
                if (!chunk_gpu_ready(field, store, make_chunk_key(child_lod, child_base_x + x, child_base_y + y, child_base_z + z))) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool should_draw_gpu_lod_chunk(const MineSdfField& field, const GpuChunkStore& store, MortonChunkKey key) {
    if (!chunk_gpu_ready(field, store, key)) {
        return false;
    }
    if (key.lod > 0u) {
        const MortonChunkKey parent = parent_chunk_key(key);
        if (chunk_gpu_ready(field, store, parent) && !all_child_chunks_gpu_ready(field, store, parent)) {
            return false;
        }
    }
    if (all_child_chunks_gpu_ready(field, store, key)) {
        return false;
    }
    return true;
}

std::vector<MortonChunkKey> collect_gpu_visible_lod_keys(MineSdfField& field, const GpuChunkStore& store) {
    field.visible_lod_counts = {0u, 0u, 0u};
    std::vector<MortonChunkKey> visible_keys;
    visible_keys.reserve(field.active_keys.size());
    for (const MortonChunkKey& key : field.active_keys) {
        if (!should_draw_gpu_lod_chunk(field, store, key)) {
            continue;
        }
        visible_keys.push_back(key);
        if (key.lod < field.visible_lod_counts.size()) {
            ++field.visible_lod_counts[key.lod];
        }
    }
    return visible_keys;
}

float axis_value(ae::Vec3 p, uint32_t axis) {
    if (axis == 0u) return p.x;
    if (axis == 1u) return p.y;
    return p.z;
}

ae::Vec3 axis_offset(uint32_t axis, float amount) {
    if (axis == 0u) return {amount, 0.0f, 0.0f};
    if (axis == 1u) return {0.0f, amount, 0.0f};
    return {0.0f, 0.0f, amount};
}

struct VisibleLodIndex {
    std::unordered_set<uint64_t> codes;
};

uint32_t chunk_coord_for_point(uint32_t lod, float value, uint32_t chunk_size) {
    const uint32_t chunks_per_axis = lod_chunks_per_axis(lod, chunk_size);
    const int32_t coord = lod_coord(lod, value);
    return static_cast<uint32_t>(std::clamp(
        coord / static_cast<int32_t>(chunk_size),
        0,
        static_cast<int32_t>(chunks_per_axis - 1u)
    ));
}

int visible_lower_lod_at_point(const VisibleLodIndex& visible_index, const MineSdfField& field, ae::Vec3 p, MortonChunkKey current) {
    for (int32_t lod = static_cast<int32_t>(current.lod) - 1; lod >= 0; --lod) {
        const uint32_t lod_u = static_cast<uint32_t>(lod);
        const MortonChunkKey candidate = make_chunk_key(
            lod_u,
            chunk_coord_for_point(lod_u, p.x, field.chunk_size),
            chunk_coord_for_point(lod_u, p.y, field.chunk_size),
            chunk_coord_for_point(lod_u, p.z, field.chunk_size)
        );
        if (visible_index.codes.find(candidate.code) != visible_index.codes.end()) {
            return lod;
        }
    }
    return -1;
}

bool face_needs_transition(
    const MineSdfField& field,
    MortonChunkKey key,
    const VisibleLodIndex& visible_index,
    ae::Vec3 minimum,
    ae::Vec3 maximum,
    uint32_t axis,
    float sign
) {
    const ae::Vec3 center = (minimum + maximum) * 0.5f;
    const float step = (GridRadius * 2.0f) / static_cast<float>(lod_resolution(key.lod));
    ae::Vec3 probe = center;
    if (axis == 0u) probe.x = sign < 0.0f ? minimum.x : maximum.x;
    if (axis == 1u) probe.y = sign < 0.0f ? minimum.y : maximum.y;
    if (axis == 2u) probe.z = sign < 0.0f ? minimum.z : maximum.z;
    probe = probe + axis_offset(axis, sign * step * 0.65f);
    if (std::fabs(probe.x) > GridRadius || std::fabs(probe.y) > GridRadius || std::fabs(probe.z) > GridRadius) {
        return false;
    }
    const int neighbor_lod = visible_lower_lod_at_point(visible_index, field, probe, key);
    return neighbor_lod >= 0 && key.lod > static_cast<uint32_t>(neighbor_lod);
}

void append_lod_transition_strips(
    TestbedMesh& mesh,
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash>& combined_lookup,
    const MineSdfChunk& chunk,
    uint32_t chunk_size,
    const std::array<bool, 6u>& transition_faces
) {
    if (!std::any_of(transition_faces.begin(), transition_faces.end(), [](bool value) { return value; })) {
        return;
    }
    ae::Vec3 minimum;
    ae::Vec3 maximum;
    chunk_world_bounds(chunk_size, chunk.key, minimum, maximum);
    const float step = (GridRadius * 2.0f) / static_cast<float>(lod_resolution(chunk.key.lod));
    const float epsilon = step * 0.035f;
    const float skirt = step * 0.70f;
    auto on_face = [&](ae::Vec3 p, uint32_t face) {
        const uint32_t axis = face / 2u;
        const bool min_face = (face % 2u) == 0u;
        const float plane = axis_value(min_face ? minimum : maximum, axis);
        return std::fabs(axis_value(p, axis) - plane) <= epsilon;
    };
    auto skirt_offset = [&](uint32_t face) {
        const uint32_t axis = face / 2u;
        const float sign = (face % 2u) == 0u ? -1.0f : 1.0f;
        return axis_offset(axis, sign * skirt);
    };
    auto add_transition_vertex = [&](ae::Vec3 p, ae::Vec3 normal) {
        return add_vertex(mesh, combined_lookup, p, normal, ae::Vec3{0.20f, 0.15f, 0.10f});
    };
    auto append_transition_triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
        if (a == b || b == c || c == a) {
            return;
        }
        const ae::Vec3 pa = mesh.vertices[a].position;
        const ae::Vec3 pb = mesh.vertices[b].position;
        const ae::Vec3 pc = mesh.vertices[c].position;
        const ae::Vec3 face = ae::cross(pb - pa, pc - pa);
        if (ae::length(face) <= 0.0000008f) {
            return;
        }
        ae::Vec3 average_normal = ae::normalize(mesh.vertices[a].normal + mesh.vertices[b].normal + mesh.vertices[c].normal);
        if (ae::length(average_normal) <= 0.000001f) {
            average_normal = ae::normalize(face);
        }
        if (ae::dot(face, average_normal) < 0.0f) {
            std::swap(b, c);
        }
        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
        mesh.indices.push_back(c);
    };
    for (uint32_t i = 0; i + 2u < chunk.indices.size(); i += 3u) {
        const std::array<uint32_t, 3u> tri = {chunk.indices[i], chunk.indices[i + 1u], chunk.indices[i + 2u]};
        for (uint32_t edge = 0; edge < 3u; ++edge) {
            const uint32_t ia = tri[edge];
            const uint32_t ib = tri[(edge + 1u) % 3u];
            if (ia >= chunk.vertices.size() || ib >= chunk.vertices.size()) {
                continue;
            }
            const RenderVertex& va = chunk.vertices[ia];
            const RenderVertex& vb = chunk.vertices[ib];
            for (uint32_t face = 0; face < 6u; ++face) {
                if (!transition_faces[face] || !on_face(va.position, face) || !on_face(vb.position, face)) {
                    continue;
                }
                const ae::Vec3 offset = skirt_offset(face);
                const uint32_t a = add_transition_vertex(va.position, va.normal);
                const uint32_t b = add_transition_vertex(vb.position, vb.normal);
                const uint32_t c = add_transition_vertex(vb.position + offset, vb.normal);
                const uint32_t d = add_transition_vertex(va.position + offset, va.normal);
                append_transition_triangle(a, b, c);
                append_transition_triangle(a, c, d);
            }
        }
    }
}

TestbedMesh remesh_dirty_chunks(MineSdfField& field, const MineSettings& settings, const MineNetwork& network) {
    const auto begin = std::chrono::steady_clock::now();
    const uint32_t dirty_before = dirty_chunk_count(field);

    TestbedMesh mesh;
    field.transition_triangle_count = 0u;
    const std::vector<MortonChunkKey> visible_keys = collect_visible_lod_keys(field);
    VisibleLodIndex visible_index;
    visible_index.codes.reserve(visible_keys.size() * 2u + 16u);
    for (MortonChunkKey key : visible_keys) {
        visible_index.codes.insert(key.code);
    }

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> combined_lookup;
    for (const MortonChunkKey& key : visible_keys) {
        const auto found = field.chunks.find(key.code);
        if (found == field.chunks.end() || found->second.state != ChunkState::Ready) {
            continue;
        }
        const MineSdfChunk& chunk = found->second;
        std::vector<uint32_t> remap(chunk.vertices.size(), UINT32_MAX);
        for (uint32_t local_index : chunk.indices) {
            if (local_index >= chunk.vertices.size()) {
                ++mesh.stats.invalid_indices;
                continue;
            }
            if (remap[local_index] == UINT32_MAX) {
                const RenderVertex& vertex = chunk.vertices[local_index];
                remap[local_index] = add_vertex(mesh, combined_lookup, vertex.position, vertex.normal, vertex.color);
            }
            mesh.indices.push_back(remap[local_index]);
        }
        mesh.stats.invalid_indices += chunk.invalid_indices;
        mesh.stats.degenerate_triangles += chunk.degenerate_triangles;

        ae::Vec3 minimum;
        ae::Vec3 maximum;
        chunk_world_bounds(field.chunk_size, key, minimum, maximum);
        const std::array<bool, 6u> transition_faces = {{
            face_needs_transition(field, key, visible_index, minimum, maximum, 0u, -1.0f),
            face_needs_transition(field, key, visible_index, minimum, maximum, 0u, 1.0f),
            face_needs_transition(field, key, visible_index, minimum, maximum, 1u, -1.0f),
            face_needs_transition(field, key, visible_index, minimum, maximum, 1u, 1.0f),
            face_needs_transition(field, key, visible_index, minimum, maximum, 2u, -1.0f),
            face_needs_transition(field, key, visible_index, minimum, maximum, 2u, 1.0f),
        }};
        const uint32_t before = static_cast<uint32_t>(mesh.indices.size() / 3u);
        append_lod_transition_strips(mesh, combined_lookup, chunk, field.chunk_size, transition_faces);
        field.transition_triangle_count += static_cast<uint32_t>(mesh.indices.size() / 3u) - before;
    }

    mesh.stats.seed = settings.seed;
    mesh.stats.entrances = static_cast<uint32_t>(network.entrances.size());
    mesh.stats.primary_shafts = static_cast<uint32_t>(network.primary_shafts.size());
    mesh.stats.branch_tunnels = static_cast<uint32_t>(network.branches.size());
    mesh.stats.chambers = static_cast<uint32_t>(network.chambers.size());
    mesh.stats.grid_resolution = field.resolution;
    mesh.stats.vertices = static_cast<uint32_t>(mesh.vertices.size());
    mesh.stats.triangles = static_cast<uint32_t>(mesh.indices.size() / 3u);
    mesh.stats.ok = mesh.stats.invalid_indices == 0u && mesh.stats.degenerate_triangles == 0u;
    mesh.stats.edit_count = static_cast<uint32_t>(field.edits.size());
    mesh.stats.dirty_chunks = dirty_before;
    mesh.stats.brush_radius = field.brush_radius;
    mesh.stats.last_remesh_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    field.last_remesh_ms = mesh.stats.last_remesh_ms;
    std::ostringstream message;
    message << (mesh.stats.ok ? "Mesh validation OK" : "Mesh validation FAILED")
            << ": seed " << mesh.stats.seed
            << ", entrances " << mesh.stats.entrances
            << ", primary shafts " << mesh.stats.primary_shafts
            << ", branches " << mesh.stats.branch_tunnels
            << ", chambers " << mesh.stats.chambers
            << ", edits " << mesh.stats.edit_count
            << ", dirty chunks " << mesh.stats.dirty_chunks
            << ", brush " << mesh.stats.brush_radius
            << ", remesh " << mesh.stats.last_remesh_ms << " ms"
            << ", grid " << mesh.stats.grid_resolution
            << ", vertices " << mesh.stats.vertices
            << ", triangles " << mesh.stats.triangles
            << ", invalid " << mesh.stats.invalid_indices
            << ", degenerate " << mesh.stats.degenerate_triangles
            << ", open " << mesh.stats.open_edges
            << ", nonmanifold " << mesh.stats.nonmanifold_edges;
    mesh.stats.message = message.str();
    return mesh;
}

BuildStats update_streaming_stats(
    MineSdfField& field,
    const MineSettings& settings,
    const MineNetwork& network,
    const std::vector<MortonChunkKey>& visible_keys
) {
    const auto begin = std::chrono::steady_clock::now();
    BuildStats stats;
    field.transition_triangle_count = 0u;
    for (MortonChunkKey key : visible_keys) {
        const auto found = field.chunks.find(key.code);
        if (found == field.chunks.end() || found->second.state != ChunkState::Ready) {
            continue;
        }
        const MineSdfChunk& chunk = found->second;
        stats.vertices += static_cast<uint32_t>(chunk.vertices.size());
        stats.triangles += static_cast<uint32_t>(chunk.indices.size() / 3u);
        stats.invalid_indices += chunk.invalid_indices;
        stats.degenerate_triangles += chunk.degenerate_triangles;
    }
    stats.seed = settings.seed;
    stats.entrances = static_cast<uint32_t>(network.entrances.size());
    stats.primary_shafts = static_cast<uint32_t>(network.primary_shafts.size());
    stats.branch_tunnels = static_cast<uint32_t>(network.branches.size());
    stats.chambers = static_cast<uint32_t>(network.chambers.size());
    stats.grid_resolution = field.resolution;
    stats.ok = stats.invalid_indices == 0u && stats.degenerate_triangles == 0u;
    stats.edit_count = static_cast<uint32_t>(field.edits.size());
    stats.dirty_chunks = dirty_chunk_count(field);
    stats.brush_radius = field.brush_radius;
    stats.last_remesh_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    field.last_remesh_ms = stats.last_remesh_ms;
    std::ostringstream message;
    message << (stats.ok ? "Streaming mesh OK" : "Streaming mesh CHECK")
            << ": seed " << stats.seed
            << ", visible chunks " << visible_keys.size()
            << ", grid " << stats.grid_resolution
            << ", vertices " << stats.vertices
            << ", triangles " << stats.triangles
            << ", invalid " << stats.invalid_indices
            << ", degenerate " << stats.degenerate_triangles;
    stats.message = message.str();
    return stats;
}

void start_worker_pool(WorkerPool& pool) {
    pool.worker_count = std::max(1u, std::thread::hardware_concurrency() > 2u ? std::thread::hardware_concurrency() - 1u : 1u);
    for (uint32_t i = 0; i < pool.worker_count; ++i) {
        pool.threads.emplace_back([&pool]() {
            while (!pool.stop.load()) {
                WorkerTask task;
                {
                    std::unique_lock<std::mutex> lock(pool.mutex);
                    pool.cv.wait(lock, [&]() {
                        return pool.stop.load() || !pool.queue.empty();
                    });
                    if (pool.stop.load()) {
                        return;
                    }
                    task = std::move(const_cast<WorkerTask&>(pool.queue.top()));
                    pool.queue.pop();
                }
                ChunkBuildResult result = task.build();
                {
                    std::lock_guard<std::mutex> lock(pool.mutex);
                    pool.completed.push_back(std::move(result));
                }
            }
        });
    }
}

void stop_worker_pool(WorkerPool& pool) {
    pool.stop.store(true);
    pool.cv.notify_all();
    for (std::thread& thread : pool.threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

uint32_t worker_queue_size(WorkerPool& pool) {
    std::lock_guard<std::mutex> lock(pool.mutex);
    return static_cast<uint32_t>(pool.queue.size());
}

void clear_worker_queue(WorkerPool& pool) {
    std::lock_guard<std::mutex> lock(pool.mutex);
    while (!pool.queue.empty()) {
        pool.queue.pop();
    }
    pool.completed.clear();
}

void enqueue_chunk_build(WorkerPool& pool, std::shared_ptr<ChunkBuildContext> context, MortonChunkKey key, uint32_t build_serial, float priority) {
    WorkerTask task;
    task.generation = context->generation;
    task.priority = priority;
    task.build = [context, key, build_serial]() {
        return build_chunk_result(*context, key, build_serial);
    };
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        task.sequence = pool.next_sequence++;
        pool.queue.push(std::move(task));
    }
    pool.cv.notify_one();
}

float chunk_priority(const MineSdfField& field, MortonChunkKey key, ae::Vec3 focus) {
    ae::Vec3 minimum;
    ae::Vec3 maximum;
    chunk_world_bounds(field.chunk_size, key, minimum, maximum);
    const ae::Vec3 center = (minimum + maximum) * 0.5f;
    const float lod_bias = key.lod == 0u ? -0.55f : key.lod == 1u ? -0.18f : -0.04f;
    const auto found = field.chunks.find(key.code);
    const float urgent_bias = found != field.chunks.end() && found->second.urgent ? -4.0f : 0.0f;
    return urgent_bias + lod_bias + ae::length(center - focus) + static_cast<float>(key.code & 0xffffu) * 0.0000001f;
}

uint32_t queue_dirty_chunks(MineSdfField& field, WorkerPool& pool, ae::Vec3 focus, uint32_t max_jobs = UINT32_MAX) {
    if (field.streaming_paused) {
        return 0;
    }
    auto context = std::make_shared<ChunkBuildContext>();
    context->generation = field.generation;
    context->resolution = field.resolution;
    context->chunk_size = field.chunk_size;
    context->procedural_primitives = field.procedural_primitives;
    context->edit_primitives = field.edit_primitives;

    std::vector<std::pair<float, MortonChunkKey>> jobs;
    jobs.reserve(field.active_keys.size());
    for (const MortonChunkKey& key : field.active_keys) {
        auto found = field.chunks.find(key.code);
        if (found == field.chunks.end()) {
            continue;
        }
        MineSdfChunk& chunk = found->second;
        if (!chunk.dirty || chunk.queued) {
            continue;
        }
        jobs.push_back({chunk_priority(field, key, focus), key});
    }
    std::sort(jobs.begin(), jobs.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    uint32_t queued = 0;
    for (const auto& job : jobs) {
        if (queued >= max_jobs) {
            break;
        }
        MineSdfChunk& chunk = field.chunks[job.second.code];
        chunk.queued = true;
        chunk.state = ChunkState::Queued;
        chunk.priority = job.first;
        enqueue_chunk_build(pool, context, job.second, chunk.build_serial, job.first);
        ++queued;
    }
    field.queued_jobs_last_frame = queued;
    return queued;
}

uint32_t drain_completed_chunks(MineSdfField& field, WorkerPool& pool, uint32_t max_results = 128u) {
    std::vector<ChunkBuildResult> completed;
    {
        std::lock_guard<std::mutex> lock(pool.mutex);
        const uint32_t count = std::min<uint32_t>(max_results, static_cast<uint32_t>(pool.completed.size()));
        completed.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            completed.push_back(std::move(pool.completed.back()));
            pool.completed.pop_back();
        }
    }

    uint32_t accepted = 0;
    for (ChunkBuildResult& result : completed) {
        if (result.generation != field.generation) {
            ++field.stale_result_drops;
            continue;
        }
        auto found = field.chunks.find(result.key.code);
        if (found == field.chunks.end()) {
            ++field.stale_result_drops;
            continue;
        }
        if (result.build_serial != found->second.build_serial) {
            ++field.stale_result_drops;
            continue;
        }
        result.chunk.priority = found->second.priority;
        result.chunk.urgent = found->second.urgent;
        result.chunk.gpu_dirty = true;
        field.chunks[result.key.code] = std::move(result.chunk);
        ++accepted;
    }
    field.completed_jobs_last_frame = accepted;
    field.completed_jobs_total += accepted;
    return accepted;
}

void append_line(std::vector<RenderVertex>& lines, ae::Vec3 a, ae::Vec3 b, ae::Vec3 color) {
    lines.push_back({a, ae::normalize(a), color});
    lines.push_back({b, ae::normalize(b), color});
}

void append_sphere_preview(std::vector<RenderVertex>& lines, ae::Vec3 center, float radius, ae::Vec3 color) {
    constexpr uint32_t Segments = 32u;
    for (uint32_t i = 0; i < Segments; ++i) {
        const float a0 = static_cast<float>(i) * ae::Pi * 2.0f / static_cast<float>(Segments);
        const float a1 = static_cast<float>(i + 1u) * ae::Pi * 2.0f / static_cast<float>(Segments);
        append_line(lines, center + ae::Vec3{std::cos(a0) * radius, std::sin(a0) * radius, 0.0f}, center + ae::Vec3{std::cos(a1) * radius, std::sin(a1) * radius, 0.0f}, color);
        append_line(lines, center + ae::Vec3{std::cos(a0) * radius, 0.0f, std::sin(a0) * radius}, center + ae::Vec3{std::cos(a1) * radius, 0.0f, std::sin(a1) * radius}, color);
        append_line(lines, center + ae::Vec3{0.0f, std::cos(a0) * radius, std::sin(a0) * radius}, center + ae::Vec3{0.0f, std::cos(a1) * radius, std::sin(a1) * radius}, color);
    }
}

std::vector<RenderVertex> build_graph_lines(const MineNetwork& network, const MineSdfField& field, bool has_preview, ae::Vec3 preview_center) {
    std::vector<RenderVertex> lines;
    for (const MinePath& path : network.primary_shafts) {
        for (uint32_t i = 0; i + 1u < path.points.size(); ++i) {
            append_line(lines, path.points[i], path.points[i + 1u], color_primary_line());
        }
    }
    for (const MinePath& path : network.branches) {
        for (uint32_t i = 0; i + 1u < path.points.size(); ++i) {
            append_line(lines, path.points[i], path.points[i + 1u], color_branch_line());
        }
    }
    for (const MineChamber& chamber : network.chambers) {
        const ae::Vec3 color = chamber.resource ? color_resource() : color_chamber_line();
        const float r = chamber.radius * 0.72f;
        append_line(lines, chamber.center + ae::Vec3{r, 0.0f, 0.0f}, chamber.center - ae::Vec3{r, 0.0f, 0.0f}, color);
        append_line(lines, chamber.center + ae::Vec3{0.0f, r, 0.0f}, chamber.center - ae::Vec3{0.0f, r, 0.0f}, color);
        append_line(lines, chamber.center + ae::Vec3{0.0f, 0.0f, r}, chamber.center - ae::Vec3{0.0f, 0.0f, r}, color);
    }
    for (const EditableMineObject& object : field.objects) {
        if (!object.enabled || object.points.empty()) {
            continue;
        }
        const ae::Vec3 color = object.kind == EditableObjectKind::Chamber ? color_chamber_line() : color_branch_line();
        if (object.kind == EditableObjectKind::Chamber) {
            append_sphere_preview(lines, object.points.front(), object.radius, color);
        } else {
            for (uint32_t i = 0; i + 1u < object.points.size(); ++i) {
                append_line(lines, object.points[i], object.points[i + 1u], color);
            }
        }
    }
    if (field.has_tunnel_mark) {
        append_sphere_preview(lines, field.tunnel_mark, field.brush_radius * 0.65f, color_primary_line());
    }
    if (has_preview) {
        append_sphere_preview(lines, preview_center, field.brush_radius, color_brush_line());
    }
    return lines;
}

void upload_lines(GpuMesh& gpu, const std::vector<RenderVertex>& lines) {
    if (gpu.line_vao == 0) {
        glGenVertexArrays(1, &gpu.line_vao);
        glGenBuffers(1, &gpu.line_vbo);
    }
    glBindVertexArray(gpu.line_vao);
    glBindBuffer(GL_ARRAY_BUFFER, gpu.line_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(lines.size() * sizeof(RenderVertex)), lines.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(offsetof(RenderVertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(offsetof(RenderVertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(offsetof(RenderVertex, color)));
    gpu.line_vertex_count = static_cast<uint32_t>(lines.size());
}

void configure_surface_vertex_attributes() {
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(offsetof(RenderVertex, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(offsetof(RenderVertex, normal)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(offsetof(RenderVertex, color)));
}

void destroy_gpu_chunk(GpuChunk& chunk) {
    if (chunk.ebo != 0) glDeleteBuffers(1, &chunk.ebo);
    if (chunk.vbo != 0) glDeleteBuffers(1, &chunk.vbo);
    if (chunk.vao != 0) glDeleteVertexArrays(1, &chunk.vao);
    chunk = {};
}

void clear_gpu_chunks(GpuChunkStore& store) {
    for (auto& entry : store.chunks) {
        destroy_gpu_chunk(entry.second);
    }
    store.chunks.clear();
    store.urgent_uploads.clear();
    store.pending_uploads.clear();
    store.urgent_codes.clear();
    store.pending_codes.clear();
}

void upload_chunk(GpuChunkStore& store, const MineSdfChunk& chunk, uint32_t generation) {
    GpuChunk& gpu_chunk = store.chunks[chunk.key.code];
    if (gpu_chunk.vao == 0) {
        glGenVertexArrays(1, &gpu_chunk.vao);
        glGenBuffers(1, &gpu_chunk.vbo);
        glGenBuffers(1, &gpu_chunk.ebo);
    }
    glBindVertexArray(gpu_chunk.vao);
    glBindBuffer(GL_ARRAY_BUFFER, gpu_chunk.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(chunk.vertices.size() * sizeof(RenderVertex)), chunk.vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpu_chunk.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(chunk.indices.size() * sizeof(uint32_t)), chunk.indices.data(), GL_STATIC_DRAW);
    configure_surface_vertex_attributes();
    gpu_chunk.index_count = static_cast<uint32_t>(chunk.indices.size());
    gpu_chunk.generation = generation;
}

void enqueue_visible_gpu_uploads(MineSdfField& field, GpuChunkStore& store, const std::vector<MortonChunkKey>& visible_keys) {
    for (MortonChunkKey key : visible_keys) {
        auto found = field.chunks.find(key.code);
        if (found == field.chunks.end() || found->second.state != ChunkState::Ready || !found->second.gpu_dirty) {
            continue;
        }
        if (found->second.urgent) {
            if (store.urgent_codes.insert(key.code).second) {
                store.urgent_uploads.push_back(key.code);
            }
        } else if (store.pending_codes.insert(key.code).second) {
            store.pending_uploads.push_back(key.code);
        }
    }
}

uint32_t upload_queued_chunks(MineSdfField& field, GpuChunkStore& store, uint32_t max_uploads) {
    uint32_t uploaded = 0;
    auto pop_code = [&]() {
        if (!store.urgent_uploads.empty()) {
            const uint64_t code = store.urgent_uploads.front();
            store.urgent_uploads.pop_front();
            store.urgent_codes.erase(code);
            return code;
        }
        const uint64_t code = store.pending_uploads.front();
        store.pending_uploads.pop_front();
        store.pending_codes.erase(code);
        return code;
    };
    while (uploaded < max_uploads && (!store.urgent_uploads.empty() || !store.pending_uploads.empty())) {
        const uint64_t code = pop_code();
        auto found = field.chunks.find(code);
        if (found == field.chunks.end() || found->second.state != ChunkState::Ready) {
            continue;
        }
        upload_chunk(store, found->second, field.generation);
        found->second.gpu_dirty = false;
        found->second.urgent = false;
        ++uploaded;
    }
    field.uploaded_chunks_last_frame = uploaded;
    return uploaded;
}

uint32_t draw_visible_chunks(const GpuChunkStore& store, const std::vector<MortonChunkKey>& visible_keys) {
    uint32_t index_count = 0;
    for (MortonChunkKey key : visible_keys) {
        const auto found = store.chunks.find(key.code);
        if (found == store.chunks.end() || found->second.vao == 0 || found->second.index_count == 0u) {
            continue;
        }
        glBindVertexArray(found->second.vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(found->second.index_count), GL_UNSIGNED_INT, nullptr);
        index_count += found->second.index_count;
    }
    return index_count;
}

void destroy_mesh(GpuMesh& gpu) {
    if (gpu.line_vbo != 0) glDeleteBuffers(1, &gpu.line_vbo);
    if (gpu.line_vao != 0) glDeleteVertexArrays(1, &gpu.line_vao);
    gpu = {};
}

ae::Vec3 rotate_axis_angle(ae::Vec3 value, ae::Vec3 axis, float radians) {
    axis = ae::normalize(axis);
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return value * c + ae::cross(axis, value) * s + axis * (ae::dot(axis, value) * (1.0f - c));
}

void orthonormalize(Camera& camera) {
    camera.forward = ae::normalize(camera.forward);
    camera.right = camera.right - camera.forward * ae::dot(camera.right, camera.forward);
    if (ae::length(camera.right) <= 0.00001f) {
        camera.right = ae::normalize(ae::cross(camera.forward, {0.0f, 1.0f, 0.0f}));
        if (ae::length(camera.right) <= 0.00001f) {
            camera.right = {1.0f, 0.0f, 0.0f};
        }
    } else {
        camera.right = ae::normalize(camera.right);
    }
    camera.up = ae::normalize(ae::cross(camera.right, camera.forward));
}

void rotate_fly_camera(Camera& camera, float yaw_delta, float pitch_delta, float roll_delta) {
    if (yaw_delta != 0.0f) {
        camera.forward = rotate_axis_angle(camera.forward, camera.up, yaw_delta);
        camera.right = rotate_axis_angle(camera.right, camera.up, yaw_delta);
    }
    if (pitch_delta != 0.0f) {
        camera.forward = rotate_axis_angle(camera.forward, camera.right, pitch_delta);
        camera.up = rotate_axis_angle(camera.up, camera.right, pitch_delta);
    }
    if (roll_delta != 0.0f) {
        camera.right = rotate_axis_angle(camera.right, camera.forward, roll_delta);
        camera.up = rotate_axis_angle(camera.up, camera.forward, roll_delta);
    }
    orthonormalize(camera);
}

ae::Vec3 fly_forward(const Camera& camera) {
    return camera.forward;
}

void fly_axes(const Camera& camera, ae::Vec3& forward, ae::Vec3& right, ae::Vec3& up) {
    forward = camera.forward;
    right = camera.right;
    up = camera.up;
}

ae::Vec3 fly_up(const Camera& camera) {
    return camera.up;
}

ae::Vec3 orbit_eye(const Camera& camera) {
    const float cp = std::cos(camera.orbit_pitch);
    return {
        std::sin(camera.orbit_yaw) * cp * camera.orbit_distance,
        std::sin(camera.orbit_pitch) * camera.orbit_distance,
        std::cos(camera.orbit_yaw) * cp * camera.orbit_distance,
    };
}

void aim_camera(Camera& camera, ae::Vec3 target) {
    camera.forward = ae::normalize(target - camera.position);
    camera.right = ae::cross(camera.forward, {0.0f, 1.0f, 0.0f});
    if (ae::length(camera.right) <= 0.00001f) {
        camera.right = {1.0f, 0.0f, 0.0f};
    }
    orthonormalize(camera);
}

bool raycast_field(const MineSdfField& field, ae::Vec3 origin, ae::Vec3 direction, ae::Vec3& hit) {
    direction = ae::normalize(direction);
    float previous_t = 0.0f;
    float previous_value = sample_field(field, origin);
    float best_abs = std::fabs(previous_value);
    ae::Vec3 best = origin;
    constexpr float MaxDistance = 2.5f;
    const float step = std::max(0.006f, field.voxel * 0.55f);
    for (float t = step; t <= MaxDistance; t += step) {
        const ae::Vec3 p = origin + direction * t;
        if (ae::length(p) > GridRadius + 0.08f) {
            break;
        }
        const float value = sample_field(field, p);
        if (std::fabs(value) < best_abs) {
            best_abs = std::fabs(value);
            best = p;
        }
        if ((previous_value <= 0.0f && value >= 0.0f) || (previous_value >= 0.0f && value <= 0.0f)) {
            float lo = previous_t;
            float hi = t;
            for (uint32_t i = 0; i < 8u; ++i) {
                const float mid = (lo + hi) * 0.5f;
                const float mid_value = sample_field(field, origin + direction * mid);
                if ((previous_value <= 0.0f && mid_value >= 0.0f) || (previous_value >= 0.0f && mid_value <= 0.0f)) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hit = origin + direction * ((lo + hi) * 0.5f);
            return true;
        }
        previous_t = t;
        previous_value = value;
    }
    if (best_abs < field.brush_radius * 1.6f) {
        hit = best;
        return true;
    }
    hit = origin + direction * 0.35f;
    return false;
}

EditableMineObject* find_object(MineSdfField& field, uint32_t object_id) {
    for (EditableMineObject& object : field.objects) {
        if (object.id == object_id) {
            return &object;
        }
    }
    return nullptr;
}

EditableMineObject* nearest_enabled_object(MineSdfField& field, ae::Vec3 p) {
    EditableMineObject* best = nullptr;
    float best_distance = std::numeric_limits<float>::max();
    for (EditableMineObject& object : field.objects) {
        if (!object.enabled || object.points.empty()) {
            continue;
        }
        float distance = std::numeric_limits<float>::max();
        if (object.kind == EditableObjectKind::Chamber) {
            distance = ae::length(p - object.points.front()) - object.radius;
        } else {
            for (uint32_t i = 0; i + 1u < object.points.size(); ++i) {
                distance = std::min(distance, distance_to_segment(p, object.points[i], object.points[i + 1u]) - object.radius);
            }
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = &object;
        }
    }
    return best;
}

void place_camera_in_first_shaft(Camera& camera, const MineNetwork& network) {
    if (network.primary_shafts.empty() || network.primary_shafts.front().points.size() < 4u) {
        return;
    }
    camera.position = path_sample(network.primary_shafts.front(), 0.18f);
    aim_camera(camera, path_sample(network.primary_shafts.front(), 0.62f));
}

void update_window_title(
    SDL_Window* window,
    const MineSettings& settings,
    const BuildStats& stats,
    bool fly_mode,
    bool cutaway,
    const MineSdfField* field = nullptr,
    uint32_t queued_jobs = 0,
    uint32_t worker_count = 0
) {
    std::ostringstream title;
    title << "AETHERONUS | seed " << settings.seed
          << " mines " << settings.mine_density
          << " radius " << settings.tunnel_radius
          << " grid " << stats.grid_resolution
          << " fps " << (field != nullptr ? static_cast<uint32_t>(std::round(field->fps_estimate)) : 0u)
          << " edits " << stats.edit_count
          << " dirty " << stats.dirty_chunks
          << " brush " << stats.brush_radius
          << " remesh " << stats.last_remesh_ms << "ms"
          << " chunks " << (field != nullptr ? field->chunks.size() : 0u)
          << " active " << (field != nullptr ? field->active_keys.size() : 0u)
          << " lod " << (field != nullptr ? field->visible_lod_counts[0] : 0u)
          << "/" << (field != nullptr ? field->visible_lod_counts[1] : 0u)
          << "/" << (field != nullptr ? field->visible_lod_counts[2] : 0u)
          << " queued " << queued_jobs
          << " done " << (field != nullptr ? field->completed_jobs_total : 0u)
          << " upload " << (field != nullptr ? field->uploaded_chunks_last_frame : 0u)
          << " trans " << (field != nullptr ? field->transition_triangle_count : 0u)
          << " stale " << (field != nullptr ? field->stale_result_drops : 0u)
          << " workers " << worker_count
          << " | " << (stats.ok ? "OK" : "CHECK")
          << " open " << stats.open_edges
          << " nonmanifold " << stats.nonmanifold_edges
          << " tris " << stats.triangles
          << " | " << (fly_mode ? "fly" : "orbit")
          << (cutaway ? " cutaway" : "");
    SDL_SetWindowTitle(window, title.str().c_str());
}

} // namespace

int main(int, char**) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* window = SDL_CreateWindow(
        "AETHERONUS",
        1280,
        800,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (window == nullptr) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_GLContext context = SDL_GL_CreateContext(window);
    if (context == nullptr) {
        std::cerr << "SDL_GL_CreateContext failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
        std::cerr << "gladLoadGLLoader failed" << std::endl;
        SDL_GL_DestroyContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_GL_SetSwapInterval(1);
    SDL_SetWindowRelativeMouseMode(window, true);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glClearColor(0.018f, 0.022f, 0.025f, 1.0f);

    const ae::GoldbergConfig goldberg_config{4u, 0u};
    const ae::GoldbergTopology topology = ae::build_goldberg_topology(goldberg_config);
    const ae::PointCloud points = ae::build_surface_point_cloud(topology, {16u, 16u});
    const ae::TopologyValidation topology_validation = ae::validate_topology(topology, static_cast<uint32_t>(points.size()));
    const ae::PointCloudValidation point_validation = ae::validate_point_cloud(topology, points);
    std::cout << topology_validation.message << std::endl;
    std::cout << point_validation.message << std::endl;
    if (!topology_validation.ok || !point_validation.ok) {
        std::cerr << "Base Goldberg lattice validation failed; rendering mine testbed anyway." << std::endl;
    }

    const uint32_t shader = build_shader_program();
    const int mvp_location = glGetUniformLocation(shader, "u_mvp");
    const int model_location = glGetUniformLocation(shader, "u_model");
    const int camera_location = glGetUniformLocation(shader, "u_camera_position");
    const int cutaway_location = glGetUniformLocation(shader, "u_cutaway");

    WorkerPool workers;
    start_worker_pool(workers);

    MineSettings settings;
    SdfLodPolicy lod_policy;
    uint32_t current_generation = 1u;
    MineNetwork network = build_mine_network(points, settings);
    MineSdfField field = build_base_mine_field(settings, network);
    field.generation = current_generation;
    Camera camera;
    place_camera_in_first_shaft(camera, network);
    add_focus_lod_chunks(field, camera.position, lod_policy);
    queue_dirty_chunks(field, workers, camera.position, 384u);

    std::vector<MortonChunkKey> visible_keys = collect_visible_lod_keys(field);
    std::vector<MortonChunkKey> draw_keys;
    TestbedMesh mesh;
    mesh.stats = update_streaming_stats(field, settings, network, draw_keys);
    mesh.graph_lines = build_graph_lines(network, field, false, {});
    std::cout << "Goldberg config (" << goldberg_config.m << ", " << goldberg_config.n << ")" << std::endl;
    std::cout << "Sparse SDF queued: grid " << field.resolution
              << ", active chunks " << field.active_keys.size()
              << ", workers " << workers.worker_count
              << ", queued jobs " << worker_queue_size(workers) << std::endl;

    GpuMesh gpu;
    GpuChunkStore gpu_chunks;
    upload_lines(gpu, mesh.graph_lines);
    bool running = true;
    bool wireframe = false;
    bool graph_overlay = true;
    bool morton_overlay = false;
    bool cutaway = false;
    bool edit_inputs_armed = false;
    int width = 1280;
    int height = 800;
    const auto edit_arm_begin = std::chrono::steady_clock::now();
    auto last_frame = std::chrono::steady_clock::now();
    auto last_stats_refresh = std::chrono::steady_clock::now();
    auto last_stream_log = std::chrono::steady_clock::now();
    bool pending_stats_refresh = false;
    uint32_t edit_upload_boost_frames = 0;

    auto refresh_stats = [&]() {
        mesh.stats = update_streaming_stats(field, settings, network, draw_keys);
        update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
    };

    auto rebuild = [&]() {
        const std::vector<SdfEdit> saved_edits = field.edits;
        const std::vector<EditableMineObject> saved_objects = field.objects;
        const uint32_t next_edit_id = field.next_edit_id;
        const uint32_t next_object_id = field.next_object_id;
        const float brush_radius = field.brush_radius;
        clear_worker_queue(workers);
        clear_gpu_chunks(gpu_chunks);
        ++current_generation;
        network = build_mine_network(points, settings);
        field = build_base_mine_field(settings, network);
        field.generation = current_generation;
        field.edits = saved_edits;
        field.objects = saved_objects;
        field.next_edit_id = next_edit_id;
        field.next_object_id = next_object_id;
        field.brush_radius = std::clamp(brush_radius, field.voxel * 1.25f, 0.16f);
        rebuild_field_from_history(field);
        place_camera_in_first_shaft(camera, network);
        add_focus_lod_chunks(field, camera.position, lod_policy);
        queue_dirty_chunks(field, workers, camera.position, 384u);
        visible_keys = collect_visible_lod_keys(field);
        draw_keys = collect_gpu_visible_lod_keys(field, gpu_chunks);
        pending_stats_refresh = true;
    };

    update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);

    while (running) {
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::min(0.05f, std::chrono::duration<float>(now - last_frame).count());
        last_frame = now;
        if (dt > 0.000001f) {
            const double sample_fps = 1.0 / static_cast<double>(dt);
            field.fps_estimate = field.fps_estimate <= 0.0 ? sample_fps : field.fps_estimate * 0.92 + sample_fps * 0.08;
        }

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_WINDOW_RESIZED) {
                width = std::max(1, event.window.data1);
                height = std::max(1, event.window.data2);
            } else if (event.type == SDL_EVENT_MOUSE_MOTION) {
                if (camera.fly_mode) {
                    rotate_fly_camera(
                        camera,
                        static_cast<float>(event.motion.xrel) * 0.0027f,
                        -static_cast<float>(event.motion.yrel) * 0.0027f,
                        0.0f
                    );
                } else if ((event.motion.state & SDL_BUTTON_RMASK) != 0) {
                    camera.orbit_yaw += static_cast<float>(event.motion.xrel) * 0.004f;
                    camera.orbit_pitch = std::clamp(camera.orbit_pitch - static_cast<float>(event.motion.yrel) * 0.004f, -1.48f, 1.48f);
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && edit_inputs_armed && camera.fly_mode &&
                       (event.button.button == SDL_BUTTON_LEFT || event.button.button == SDL_BUTTON_RIGHT)) {
                ae::Vec3 hit;
                raycast_field(field, camera.position, fly_forward(camera), hit);
                const bool carve = event.button.button == SDL_BUTTON_LEFT;
                apply_sdf_edit(
                    field,
                    make_sphere_edit(
                        field,
                        carve ? SdfEditMode::Carve : SdfEditMode::Fill,
                        hit,
                        field.brush_radius,
                        carve ? EditMaterial::MineWall : EditMaterial::Fill
                    )
                );
                add_focus_lod_chunks(field, hit, lod_policy);
                queue_dirty_chunks(field, workers, hit, 512u);
                edit_upload_boost_frames = 45u;
                pending_stats_refresh = true;
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                if (camera.fly_mode) {
                    field.brush_radius = std::clamp(
                        field.brush_radius * std::pow(1.10f, event.wheel.y),
                        field.voxel * 1.25f,
                        0.16f
                    );
                    mesh.stats.brush_radius = field.brush_radius;
                    update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
                } else {
                    camera.orbit_distance = std::clamp(camera.orbit_distance * std::pow(0.88f, event.wheel.y), 1.2f, 8.0f);
                }
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                switch (event.key.key) {
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    case SDLK_TAB:
                        camera.fly_mode = !camera.fly_mode;
                        SDL_SetWindowRelativeMouseMode(window, camera.fly_mode);
                        update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
                        break;
                    case SDLK_F:
                        wireframe = !wireframe;
                        break;
                    case SDLK_B:
                        graph_overlay = !graph_overlay;
                        break;
                    case SDLK_P:
                        field.streaming_paused = !field.streaming_paused;
                        update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
                        break;
                    case SDLK_M:
                        morton_overlay = !morton_overlay;
                        graph_overlay = morton_overlay || graph_overlay;
                        update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
                        break;
                    case SDLK_C:
                        cutaway = !cutaway;
                        update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
                        break;
                    case SDLK_R:
                        ++settings.seed;
                        rebuild();
                        break;
                    case SDLK_LEFTBRACKET:
                        field.brush_radius = std::max(field.voxel * 1.25f, field.brush_radius * 0.90f);
                        mesh.stats.brush_radius = field.brush_radius;
                        update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
                        break;
                    case SDLK_RIGHTBRACKET:
                        field.brush_radius = std::min(0.16f, field.brush_radius * 1.10f);
                        mesh.stats.brush_radius = field.brush_radius;
                        update_window_title(window, settings, mesh.stats, camera.fly_mode, cutaway, &field, worker_queue_size(workers), workers.worker_count);
                        break;
                    case SDLK_G: {
                        ae::Vec3 hit;
                        raycast_field(field, camera.fly_mode ? camera.position : orbit_eye(camera), camera.fly_mode ? fly_forward(camera) : ae::normalize(-orbit_eye(camera)), hit);
                        EditableMineObject object;
                        object.id = field.next_object_id++;
                        object.kind = EditableObjectKind::Chamber;
                        object.points = {hit};
                        object.radius = field.brush_radius * 1.35f;
                        object.material = EditMaterial::MineWall;
                        field.objects.push_back(object);
                        apply_sdf_edit(field, make_sphere_edit(field, SdfEditMode::Carve, hit, object.radius, object.material, object.id));
                        add_focus_lod_chunks(field, hit, lod_policy);
                        queue_dirty_chunks(field, workers, hit, 512u);
                        edit_upload_boost_frames = 45u;
                        pending_stats_refresh = true;
                        break;
                    }
                    case SDLK_T: {
                        ae::Vec3 hit;
                        raycast_field(field, camera.fly_mode ? camera.position : orbit_eye(camera), camera.fly_mode ? fly_forward(camera) : ae::normalize(-orbit_eye(camera)), hit);
                        if (!field.has_tunnel_mark) {
                            field.tunnel_mark = hit;
                            field.has_tunnel_mark = true;
                        } else {
                            EditableMineObject object;
                            object.id = field.next_object_id++;
                            object.kind = EditableObjectKind::Tunnel;
                            object.points = {field.tunnel_mark, hit};
                            object.radius = field.brush_radius * 0.90f;
                            object.material = EditMaterial::MineWall;
                            field.objects.push_back(object);
                            apply_sdf_edit(field, make_capsule_edit(field, SdfEditMode::Carve, object.points[0], object.points[1], object.radius, object.material, object.id));
                            field.tunnel_mark = hit;
                            add_focus_lod_chunks(field, hit, lod_policy);
                            queue_dirty_chunks(field, workers, hit, 512u);
                            edit_upload_boost_frames = 45u;
                            pending_stats_refresh = true;
                        }
                        break;
                    }
                    case SDLK_X: {
                        ae::Vec3 hit;
                        raycast_field(field, camera.fly_mode ? camera.position : orbit_eye(camera), camera.fly_mode ? fly_forward(camera) : ae::normalize(-orbit_eye(camera)), hit);
                        EditableMineObject* object = nearest_enabled_object(field, hit);
                        if (object != nullptr) {
                            if (object->kind == EditableObjectKind::Chamber) {
                                apply_sdf_edit(field, make_sphere_edit(field, SdfEditMode::Fill, object->points.front(), object->radius * 1.08f, EditMaterial::Fill, object->id));
                            } else if (object->points.size() >= 2u) {
                                apply_sdf_edit(field, make_capsule_edit(field, SdfEditMode::Fill, object->points[0], object->points[1], object->radius * 1.08f, EditMaterial::Fill, object->id));
                            }
                            add_focus_lod_chunks(field, hit, lod_policy);
                            queue_dirty_chunks(field, workers, hit, 512u);
                            edit_upload_boost_frames = 45u;
                            pending_stats_refresh = true;
                        }
                        break;
                    }
                    case SDLK_U:
                        if (undo_edit(field)) {
                            const ae::Vec3 focus = camera.fly_mode ? camera.position : orbit_eye(camera);
                            add_focus_lod_chunks(field, focus, lod_policy);
                            queue_dirty_chunks(field, workers, focus, 512u);
                            edit_upload_boost_frames = 45u;
                            pending_stats_refresh = true;
                        }
                        break;
                    case SDLK_Y:
                        if (redo_edit(field)) {
                            const ae::Vec3 focus = camera.fly_mode ? camera.position : orbit_eye(camera);
                            add_focus_lod_chunks(field, focus, lod_policy);
                            queue_dirty_chunks(field, workers, focus, 512u);
                            edit_upload_boost_frames = 45u;
                            pending_stats_refresh = true;
                        }
                        break;
                    case SDLK_1:
                        settings.mine_density = settings.mine_density > 4u ? settings.mine_density - 4u : 4u;
                        rebuild();
                        break;
                    case SDLK_2:
                        settings.mine_density = std::min(64u, settings.mine_density + 4u);
                        rebuild();
                        break;
                    case SDLK_3:
                        settings.tunnel_radius = std::max(0.022f, settings.tunnel_radius * 0.90f);
                        rebuild();
                        break;
                    case SDLK_4:
                        settings.tunnel_radius = std::min(0.090f, settings.tunnel_radius * 1.10f);
                        rebuild();
                        break;
                    default:
                        break;
                }
            }
        }

        if (!edit_inputs_armed) {
            float mouse_x = 0.0f;
            float mouse_y = 0.0f;
            const bool no_mouse_buttons = SDL_GetMouseState(&mouse_x, &mouse_y) == 0u;
            const bool settled = std::chrono::duration<float>(now - edit_arm_begin).count() > 1.0f;
            edit_inputs_armed = settled && no_mouse_buttons;
        }

        int key_count = 0;
        const bool* keys = SDL_GetKeyboardState(&key_count);
        auto key_down = [&](SDL_Scancode scancode) {
            return keys != nullptr && static_cast<int>(scancode) < key_count && keys[scancode];
        };
        if (camera.fly_mode) {
            ae::Vec3 movement = {};
            ae::Vec3 forward;
            ae::Vec3 right;
            ae::Vec3 up;
            fly_axes(camera, forward, right, up);
            movement = movement + forward * (key_down(SDL_SCANCODE_W) ? 1.0f : 0.0f);
            movement = movement - forward * (key_down(SDL_SCANCODE_S) ? 1.0f : 0.0f);
            movement = movement - right * (key_down(SDL_SCANCODE_A) ? 1.0f : 0.0f);
            movement = movement + right * (key_down(SDL_SCANCODE_D) ? 1.0f : 0.0f);
            movement = movement + up * (key_down(SDL_SCANCODE_E) ? 1.0f : 0.0f);
            movement = movement - up * (key_down(SDL_SCANCODE_Q) ? 1.0f : 0.0f);
            if (ae::length(movement) > 0.000001f) {
                const float speed = key_down(SDL_SCANCODE_LSHIFT) || key_down(SDL_SCANCODE_RSHIFT) ? 0.92f : 0.32f;
                camera.position = camera.position + ae::normalize(movement) * (speed * dt);
            }
            const float roll_input = (key_down(SDL_SCANCODE_L) ? 1.0f : 0.0f) - (key_down(SDL_SCANCODE_J) ? 1.0f : 0.0f);
            if (std::fabs(roll_input) > 0.0f) {
                rotate_fly_camera(camera, 0.0f, 0.0f, roll_input * 1.85f * dt);
            }
        }

        const ae::Vec3 eye = camera.fly_mode ? camera.position : orbit_eye(camera);
        add_focus_lod_chunks(field, eye, lod_policy);
        visible_keys = collect_visible_lod_keys(field);
        const uint32_t accepted_chunks = drain_completed_chunks(field, workers, 64u);
        if (accepted_chunks > 0u) {
            pending_stats_refresh = true;
        }
        enqueue_visible_gpu_uploads(field, gpu_chunks, visible_keys);
        const uint32_t upload_budget = !gpu_chunks.urgent_uploads.empty() || edit_upload_boost_frames > 0u
            ? 12u
            : (gpu_chunks.chunks.empty() ? 4u : 1u);
        const uint32_t uploaded_chunks = upload_queued_chunks(field, gpu_chunks, upload_budget);
        if (edit_upload_boost_frames > 0u) {
            --edit_upload_boost_frames;
        }
        draw_keys = collect_gpu_visible_lod_keys(field, gpu_chunks);
        if (uploaded_chunks > 0u) {
            pending_stats_refresh = true;
        }
        const bool refresh_due = std::chrono::duration<float>(now - last_stats_refresh).count() >= 0.35f;
        if (pending_stats_refresh && refresh_due) {
            refresh_stats();
            last_stats_refresh = now;
            pending_stats_refresh = false;
        }
        if (std::chrono::duration<float>(now - last_stream_log).count() >= 2.0f) {
            last_stream_log = now;
            std::cout << "Sparse stream: completed " << field.completed_jobs_total
                      << "/" << field.active_keys.size()
                      << " chunks, queued " << worker_queue_size(workers)
                      << ", gpu chunks " << gpu_chunks.chunks.size()
                      << ", uploaded/frame " << uploaded_chunks
                      << ", visible chunks " << draw_keys.size()
                      << ", triangles " << mesh.stats.triangles
                      << ", invalid " << mesh.stats.invalid_indices
                      << ", degenerate " << mesh.stats.degenerate_triangles << std::endl;
        }
        queue_dirty_chunks(field, workers, eye, 64u);

        const ae::Vec3 target = camera.fly_mode ? camera.position + fly_forward(camera) : ae::Vec3{0.0f, 0.0f, 0.0f};
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const ae::Mat4 projection = ae::perspective(64.0f * ae::Pi / 180.0f, aspect, 0.015f, 30.0f);
        const ae::Mat4 view = ae::look_at(eye, target, camera.fly_mode ? fly_up(camera) : ae::Vec3{0.0f, 1.0f, 0.0f});
        const ae::Mat4 model = ae::identity();
        const ae::Mat4 mvp = projection * view * model;
        ae::Vec3 preview_hit;
        const bool has_preview = camera.fly_mode && raycast_field(field, eye, fly_forward(camera), preview_hit);
        mesh.graph_lines = build_graph_lines(network, field, has_preview, preview_hit);
        upload_lines(gpu, mesh.graph_lines);

        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shader);
        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, mvp.m);
        glUniformMatrix4fv(model_location, 1, GL_FALSE, model.m);
        glUniform3f(camera_location, eye.x, eye.y, eye.z);
        glUniform1i(cutaway_location, cutaway ? 1 : 0);

        glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);
        gpu.chunk_index_count = draw_visible_chunks(gpu_chunks, draw_keys);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        if (graph_overlay && gpu.line_vertex_count > 0u) {
            glDisable(GL_DEPTH_TEST);
            glLineWidth(2.0f);
            glBindVertexArray(gpu.line_vao);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(gpu.line_vertex_count));
            glEnable(GL_DEPTH_TEST);
        }

        SDL_GL_SwapWindow(window);
    }

    SDL_SetWindowRelativeMouseMode(window, false);
    stop_worker_pool(workers);
    clear_gpu_chunks(gpu_chunks);
    destroy_mesh(gpu);
    if (shader != 0) {
        glDeleteProgram(shader);
    }
    SDL_GL_DestroyContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
