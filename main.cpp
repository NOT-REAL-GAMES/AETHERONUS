#include "aetheronus/debug_renderer.hpp"
#include "aetheronus/meshing.hpp"
#include "aetheronus/planet_scale.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/spaceship.hpp"
#include "aetheronus/topology.hpp"

#include <glad/glad.h>
#include <SDL3/SDL.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t DefaultSvoDepth = 8;
constexpr uint32_t HighSvoDepth = 16;
constexpr uint32_t SvoDebugDrawDepth = 8;

enum class MeshBuildMode {
    Full,
    VoxelsOnly,
};

void print_gl_string(GLenum name, const char* label) {
    const GLubyte* value = glGetString(name);
    std::cout << label << ": " << (value != nullptr ? reinterpret_cast<const char*>(value) : "unknown") << '\n';
}

bool set_gl_attribute(SDL_GLAttr attribute, int value, const char* name) {
    if (SDL_GL_SetAttribute(attribute, value)) {
        return true;
    }

    std::cerr << "SDL_GL_SetAttribute failed for " << name << ": " << SDL_GetError() << std::endl;
    return false;
}

ae::Vec3 free_camera_forward(const ae::FreeCamera& camera) {
    const float cp = std::cos(camera.pitch);
    return {
        cp * std::sin(camera.yaw),
        std::sin(camera.pitch),
        cp * std::cos(camera.yaw),
    };
}

ae::Vec3 free_camera_right(const ae::FreeCamera& camera) {
    return ae::normalize(ae::cross(free_camera_forward(camera), {0.0f, 1.0f, 0.0f}));
}

ae::Vec3 free_camera_eye(const ae::FreeCamera& camera) {
    return camera.target - free_camera_forward(camera) * camera.distance;
}

ae::CameraView make_free_camera_view(const ae::FreeCamera& camera) {
    const ae::Vec3 forward = free_camera_forward(camera);
    return {free_camera_eye(camera), camera.target, {0.0f, 1.0f, 0.0f}};
}

ae::CameraView make_ship_follow_view(const ae::SpaceshipState& ship) {
    const ae::Vec3 ship_up = ae::normalize(ship.up);
    const ae::Vec3 forward = ae::normalize(ship.forward);
    const ae::Vec3 position = ae::spaceship_position(ship);
    return {
        position - forward * 0.70f + ship_up * 0.24f,
        position + forward * 0.55f,
        ship_up,
    };
}

bool ray_sphere_intersection(ae::Vec3 origin, ae::Vec3 direction, float radius, ae::Vec3& hit) {
    direction = ae::normalize(direction);
    const float b = 2.0f * ae::dot(origin, direction);
    const float c = ae::dot(origin, origin) - radius * radius;
    const float discriminant = b * b - 4.0f * c;
    if (discriminant < 0.0f) {
        return false;
    }

    const float root = std::sqrt(discriminant);
    const float t0 = (-b - root) * 0.5f;
    const float t1 = (-b + root) * 0.5f;
    const float t = t0 > 0.0f ? t0 : t1;
    if (t <= 0.0f) {
        return false;
    }

    hit = origin + direction * t;
    return true;
}

bool ray_triangle_intersection(
    ae::Vec3 origin,
    ae::Vec3 direction,
    ae::Vec3 a,
    ae::Vec3 b,
    ae::Vec3 c,
    float& t
) {
    constexpr float Epsilon = 0.0000001f;
    const ae::Vec3 edge_ab = b - a;
    const ae::Vec3 edge_ac = c - a;
    const ae::Vec3 p = ae::cross(direction, edge_ac);
    const float determinant = ae::dot(edge_ab, p);
    if (std::fabs(determinant) < Epsilon) {
        return false;
    }

    const float inverse_determinant = 1.0f / determinant;
    const ae::Vec3 s = origin - a;
    const float u = inverse_determinant * ae::dot(s, p);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const ae::Vec3 q = ae::cross(s, edge_ab);
    const float v = inverse_determinant * ae::dot(direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    t = inverse_determinant * ae::dot(edge_ac, q);
    return t > Epsilon;
}

bool ray_surface_net_intersection(
    const ae::SurfaceNetMesh& surface_net,
    ae::Vec3 origin_mesh,
    ae::Vec3 direction,
    ae::Vec3& hit_mesh
) {
    if (surface_net.vertices.empty() || surface_net.triangle_indices.empty()) {
        return false;
    }

    direction = ae::normalize(direction);
    bool found = false;
    float closest_t = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i + 2 < surface_net.triangle_indices.size(); i += 3) {
        const uint32_t ia = surface_net.triangle_indices[i];
        const uint32_t ib = surface_net.triangle_indices[i + 1u];
        const uint32_t ic = surface_net.triangle_indices[i + 2u];
        if (ia >= surface_net.vertices.size() || ib >= surface_net.vertices.size() || ic >= surface_net.vertices.size()) {
            continue;
        }

        float t = 0.0f;
        if (ray_triangle_intersection(
                origin_mesh,
                direction,
                surface_net.vertices[ia],
                surface_net.vertices[ib],
                surface_net.vertices[ic],
                t
            ) && t < closest_t) {
            closest_t = t;
            found = true;
        }
    }

    if (!found) {
        return false;
    }

    hit_mesh = origin_mesh + direction * closest_t;
    return true;
}

bool ray_cave_feature_intersection(
    const std::vector<ae::LocalVoxelFeature>& features,
    ae::Vec3 origin_mesh,
    ae::Vec3 direction,
    ae::Vec3& hit_mesh
) {
    direction = ae::normalize(direction);
    bool found = false;
    float closest_t = std::numeric_limits<float>::max();
    ae::Vec3 closest_hit = {};

    for (const ae::LocalVoxelFeature& feature : features) {
        if (feature.kind != ae::VoxelFeatureKind::CaveSystem) {
            continue;
        }

        const float denominator = ae::dot(direction, feature.normal_mesh);
        if (std::fabs(denominator) <= 0.000001f) {
            continue;
        }

        const float t = ae::dot(feature.center_mesh - origin_mesh, feature.normal_mesh) / denominator;
        if (t <= 0.0f || t >= closest_t) {
            continue;
        }

        const ae::Vec3 plane_hit = origin_mesh + direction * t;
        const ae::Vec3 offset = plane_hit - feature.center_mesh;
        const float u = ae::dot(offset, feature.tangent_mesh);
        const float v = ae::dot(offset, feature.bitangent_mesh);
        const float entrance_radius = ae::kilometers_to_world_units(feature.entrance_radius_km) * 1.15f;
        if ((u * u + v * v) > entrance_radius * entrance_radius) {
            continue;
        }

        const float seed_depth = std::max(
            ae::kilometers_to_world_units(2.0f),
            ae::kilometers_to_world_units(feature.tunnel_radius_km) * 0.45f
        );
        closest_hit = plane_hit - feature.normal_mesh * seed_depth;
        closest_t = t;
        found = true;
    }

    if (!found) {
        return false;
    }

    hit_mesh = closest_hit;
    return true;
}

struct MeshBuildResult {
    ae::QuantizedMesh mesh;
    ae::QuantizedMeshValidation validation;
    ae::Vec3 camera_position;
    ae::Vec3 lod_focus;
    uint32_t revision = 0;
    MeshBuildMode mode = MeshBuildMode::Full;
    std::string reason;
    double queue_wait_ms = 0.0;
    double async_build_ms = 0.0;
    double validation_ms = 0.0;
};

struct BuildProgressState {
    static constexpr double UnitsPerDone = 1000000000.0;
    std::atomic<uint64_t> units = 0;
    mutable std::mutex label_mutex;
    std::string label = "Queued";

    void set(double progress, const char* next_label) {
        const double clamped = std::clamp(progress, 0.0, 1.0);
        units.store(static_cast<uint64_t>(std::round(clamped * UnitsPerDone)), std::memory_order_relaxed);
        if (next_label != nullptr) {
            std::lock_guard lock(label_mutex);
            label = next_label;
        }
    }

    double progress() const {
        return static_cast<double>(units.load(std::memory_order_relaxed)) / UnitsPerDone;
    }

    std::string label_copy() const {
        std::lock_guard lock(label_mutex);
        return label;
    }
};

ae::QuantizedMesh make_voxel_rebuild_source(const ae::QuantizedMesh& mesh) {
    ae::QuantizedMesh result;
    result.vertices = mesh.vertices;
    result.triangle_indices = mesh.triangle_indices;
    result.line_indices = mesh.line_indices;
    result.stitch_triangle_indices = mesh.stitch_triangle_indices;
    result.stitch_line_indices = mesh.stitch_line_indices;
    result.surface_net_base_cache = mesh.surface_net_base_cache;
    result.voxel_occupancy_cache = mesh.voxel_occupancy_cache;
    result.voxel_features = mesh.voxel_features;
    result.cave_anchor_points = mesh.cave_anchor_points;
    result.triangle_count = mesh.triangle_count;
    result.rejected_triangle_count = mesh.rejected_triangle_count;
    result.stitch_triangle_count = mesh.stitch_triangle_count;
    result.boundary_edge_count = mesh.boundary_edge_count;
    result.chain_stitch_triangle_count = mesh.chain_stitch_triangle_count;
    result.fallback_stitch_triangle_count = mesh.fallback_stitch_triangle_count;
    result.boundary_run_count = mesh.boundary_run_count;
    result.paired_boundary_run_count = mesh.paired_boundary_run_count;
    result.rejected_stitch_run_count = mesh.rejected_stitch_run_count;
    result.unstitched_gap_count = mesh.unstitched_gap_count;
    result.clipped_triangle_count = mesh.clipped_triangle_count;
    result.discarded_clipped_triangle_count = mesh.discarded_clipped_triangle_count;
    result.shared_edge_path_count = mesh.shared_edge_path_count;
    result.greedy_path_step_count = mesh.greedy_path_step_count;
    result.rejected_greedy_jump_count = mesh.rejected_greedy_jump_count;
    result.cell_count = mesh.cell_count;
    result.min_cell_subdivisions = mesh.min_cell_subdivisions;
    result.max_cell_subdivisions = mesh.max_cell_subdivisions;
    result.lod_level_count = mesh.lod_level_count;
    return result;
}

MeshBuildResult build_lod_mesh_async(
    const ae::GoldbergTopology& topology,
    const ae::PointCloud& points,
    ae::MarchingCubesConfig config,
    uint32_t revision,
    std::string reason,
    std::chrono::steady_clock::time_point queued_at,
    std::shared_ptr<BuildProgressState> progress
) {
    const auto build_begin = std::chrono::steady_clock::now();
    MeshBuildResult result;
    result.camera_position = config.lod_camera_position;
    result.lod_focus = ae::normalize(config.lod_camera_position);
    result.revision = revision;
    result.mode = MeshBuildMode::Full;
    result.reason = std::move(reason);
    result.queue_wait_ms = std::chrono::duration<double, std::milli>(build_begin - queued_at).count();
    if (progress) {
        progress->set(0.0f, config.local_surface_net_depth >= HighSvoDepth ? "Starting depth-16 detail build" : "Starting mesh build");
        config.progress_callback = [progress](double value, const char* label) {
            progress->set(value, label);
        };
    }
    result.mesh = ae::build_quantized_marching_cubes(topology, points, config);
    const auto validation_begin = std::chrono::steady_clock::now();
    if (progress) {
        progress->set(0.985f, "Validating mesh");
    }
    result.validation = ae::validate_quantized_mesh(result.mesh);
    result.validation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_begin
    ).count();
    result.async_build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_begin
    ).count();
    if (progress) {
        progress->set(result.validation.ok ? 1.0f : 0.0f, result.validation.ok ? "Mesh ready" : "Mesh validation failed");
    }
    return result;
}

MeshBuildResult build_voxel_mesh_async(
    ae::QuantizedMesh mesh,
    ae::MarchingCubesConfig config,
    uint32_t revision,
    std::string reason,
    std::chrono::steady_clock::time_point queued_at,
    std::shared_ptr<BuildProgressState> progress
) {
    const auto build_begin = std::chrono::steady_clock::now();
    MeshBuildResult result;
    result.camera_position = config.lod_camera_position;
    result.lod_focus = ae::normalize(config.lod_camera_position);
    result.revision = revision;
    result.mode = MeshBuildMode::VoxelsOnly;
    result.reason = std::move(reason);
    result.queue_wait_ms = std::chrono::duration<double, std::milli>(build_begin - queued_at).count();
    if (progress) {
        progress->set(0.0f, config.local_surface_net_depth >= HighSvoDepth ? "Starting depth-16 detail rebuild" : "Starting voxel rebuild");
        config.progress_callback = [progress](double value, const char* label) {
            progress->set(value, label);
        };
    }
    result.mesh = ae::rebuild_quantized_mesh_voxels(std::move(mesh), config);
    const auto validation_begin = std::chrono::steady_clock::now();
    if (progress) {
        progress->set(0.985f, "Validating voxel rebuild");
    }
    result.validation = ae::validate_quantized_mesh(result.mesh);
    result.validation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_begin
    ).count();
    result.async_build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_begin
    ).count();
    if (progress) {
        progress->set(result.validation.ok ? 1.0f : 0.0f, result.validation.ok ? "Voxel rebuild ready" : "Voxel validation failed");
    }
    return result;
}

} // namespace

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_GL_ResetAttributes();
    if (!set_gl_attribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4, "major version") ||
        !set_gl_attribute(SDL_GL_CONTEXT_MINOR_VERSION, 3, "minor version") ||
        !set_gl_attribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE, "core profile") ||
        !set_gl_attribute(SDL_GL_DOUBLEBUFFER, 1, "double buffer") ||
        !set_gl_attribute(SDL_GL_DEPTH_SIZE, 24, "depth size")) {
        SDL_Quit();
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("AETHERONUS", 960, 540, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (window == nullptr) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (gl_context == nullptr) {
        std::cerr << "SDL_GL_CreateContext failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    if (!SDL_GL_MakeCurrent(window, gl_context)) {
        std::cerr << "SDL_GL_MakeCurrent failed: " << SDL_GetError() << std::endl;
        SDL_GL_DestroyContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress))) {
        std::cerr << "gladLoadGLLoader failed" << std::endl;
        SDL_GL_DestroyContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    if (!GLAD_GL_VERSION_4_3) {
        std::cerr << "OpenGL 4.3 is required." << std::endl;
        SDL_GL_DestroyContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    print_gl_string(GL_VENDOR, "GL_VENDOR");
    print_gl_string(GL_RENDERER, "GL_RENDERER");
    print_gl_string(GL_VERSION, "GL_VERSION");
    SDL_GL_SetSwapInterval(0);
    std::cout << "Planet scale: " << ae::PlanetCircumferenceKilometers << " km circumference, "
              << ae::PlanetRadiusKilometers << " km radius, "
              << ae::KilometersPerWorldUnit << " km/mesh-unit, rendered in kilometer-space." << std::endl;

    const ae::GoldbergTopology topology = ae::build_goldberg_topology(2);
    const ae::PointCloud points = ae::build_surface_point_cloud(topology);
    ae::FreeCamera camera;
    ae::MarchingCubesConfig mesh_config;
    mesh_config.enable_camera_proximity_lod = true;
    mesh_config.lod_min_subdivisions = 4;
    mesh_config.lod_max_subdivisions = 16;
    mesh_config.lod_levels = 4;
    mesh_config.lod_inner_patch_radius = 0.18f;
    mesh_config.lod_outer_patch_radius = 0.95f;
    mesh_config.lod_camera_position = free_camera_eye(camera);
    mesh_config.enable_fractures = false;
    mesh_config.fracture_seed = 1;
    mesh_config.fracture_gap = 0.026f;
    mesh_config.fracture_depth = 0.018f;
    mesh_config.fracture_wall_depth = 0.64f;
    mesh_config.fracture_chunk_outward_min = 0.08f;
    mesh_config.fracture_chunk_outward_max = 0.72f;
    mesh_config.svo_depth = DefaultSvoDepth;
    mesh_config.surface_net_depth = DefaultSvoDepth;
    mesh_config.local_surface_net_depth = HighSvoDepth;
    mesh_config.voxel_features.cave_depth = HighSvoDepth;
    mesh_config.svo_debug_draw_depth = SvoDebugDrawDepth;
    ae::VoxelEditSet voxel_edits;
    mesh_config.voxel_edits = voxel_edits;
    ae::MarchingCubesConfig preview_config = mesh_config;
    preview_config.enable_svo_generation = false;
    preview_config.enable_surface_net_generation = false;
    const ae::QuantizedMesh mesh = ae::build_quantized_marching_cubes(topology, points, preview_config);
    const ae::TopologyValidation validation = ae::validate_topology(topology, static_cast<uint32_t>(points.size()));
    const ae::PointCloudValidation point_validation = ae::validate_point_cloud(topology, points);
    const ae::QuantizedMeshValidation mesh_validation = ae::validate_quantized_mesh(mesh);
    std::cout << validation.message << std::endl;
    std::cout << point_validation.message << std::endl;
    std::cout << "Startup preview: " << mesh_validation.message << std::endl;
    if (!validation.ok || !point_validation.ok || !mesh_validation.ok) {
        SDL_GL_DestroyContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    ae::DebugRenderer renderer;
    if (!renderer.initialize(topology, points, mesh)) {
        SDL_GL_DestroyContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    ae::QuantizedMesh current_mesh = mesh;
    ae::Vec3 last_mesh_lod_focus = ae::normalize(mesh_config.lod_camera_position);
    ae::Vec3 requested_mesh_lod_focus = last_mesh_lod_focus;
    ae::Vec3 requested_mesh_camera_position = mesh_config.lod_camera_position;
    uint32_t mesh_revision = 1;
    uint32_t requested_mesh_revision = mesh_revision;
    std::future<MeshBuildResult> pending_mesh_build;
    MeshBuildMode requested_mesh_build_mode = MeshBuildMode::Full;
    std::string requested_mesh_build_reason = "startup";
    std::chrono::steady_clock::time_point requested_mesh_queued_at = std::chrono::steady_clock::now();
    bool mesh_build_pending = false;
    bool mesh_rebuild_requested = false;
    auto build_progress = std::make_shared<BuildProgressState>();
    bool show_fps = false;
    bool benchmark_mode = false;
    double benchmark_time = 0.0;
    ae::DebugRenderOptions render_options;
    ae::SpaceshipState spaceship;
    bool relative_mouse_enabled = false;
    float displayed_fps = 0.0f;
    uint32_t frames_since_fps_update = 0;
    std::array<double, 128> frame_times_ms = {};
    uint32_t frame_time_cursor = 0;
    double recent_worst_frame_ms = 0.0;
    uint64_t fps_update_start = SDL_GetTicksNS();
    uint64_t last_update_time = fps_update_start;
    bool window_title_shows_progress = false;
    constexpr float LodRebuildDistance = 0.12f;
    constexpr std::chrono::milliseconds NoWait{0};
    auto request_full_mesh_rebuild = [&](const char* reason) {
        requested_mesh_revision = ++mesh_revision;
        requested_mesh_lod_focus = ae::normalize(mesh_config.lod_camera_position);
        requested_mesh_camera_position = mesh_config.lod_camera_position;
        requested_mesh_build_mode = MeshBuildMode::Full;
        requested_mesh_build_reason = reason != nullptr ? reason : "full rebuild";
        requested_mesh_queued_at = std::chrono::steady_clock::now();
        mesh_rebuild_requested = true;
    };
    auto request_voxel_mesh_rebuild = [&](const char* reason) {
        requested_mesh_revision = ++mesh_revision;
        requested_mesh_lod_focus = ae::normalize(mesh_config.lod_camera_position);
        requested_mesh_camera_position = mesh_config.lod_camera_position;
        if (!mesh_rebuild_requested || requested_mesh_build_mode != MeshBuildMode::Full) {
            requested_mesh_build_mode = MeshBuildMode::VoxelsOnly;
            requested_mesh_build_reason = reason != nullptr ? reason : "voxel rebuild";
            requested_mesh_queued_at = std::chrono::steady_clock::now();
        }
        mesh_rebuild_requested = true;
    };

    build_progress->set(0.0f, "Queued startup depth-16 detail build");
    pending_mesh_build = std::async(
        std::launch::async,
        build_lod_mesh_async,
        std::cref(topology),
        std::cref(points),
        mesh_config,
        mesh_revision,
        std::string("startup"),
        requested_mesh_queued_at,
        build_progress
    );
    mesh_build_pending = true;

    bool running = true;
    while (running) {
        const auto frame_begin = std::chrono::steady_clock::now();
        float ship_mouse_yaw = 0.0f;
        float ship_mouse_pitch = 0.0f;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                if (event.key.key == SDLK_ESCAPE) {
                    running = false;
                } else if (event.key.key == SDLK_R) {
                    camera = ae::FreeCamera{};
                    mesh_config.lod_camera_position = free_camera_eye(camera);
                    const ae::Vec3 new_lod_focus = ae::normalize(mesh_config.lod_camera_position);
                    if (ae::length(new_lod_focus - requested_mesh_lod_focus) >= LodRebuildDistance) {
                        request_full_mesh_rebuild("camera reset");
                    }
                } else if (event.key.key == SDLK_F3) {
                    show_fps = !show_fps;
                } else if (event.key.key == SDLK_F2) {
                    benchmark_mode = !benchmark_mode;
                    benchmark_time = 0.0;
                    show_fps = benchmark_mode ? true : show_fps;
                    std::cout << "Benchmark camera path " << (benchmark_mode ? "enabled" : "disabled") << std::endl;
                } else if (event.key.key == SDLK_F1) {
                    render_options.show_cave_anchors = !render_options.show_cave_anchors;
                    std::cout << "Cave anchor cloud " << (render_options.show_cave_anchors ? "shown" : "hidden")
                              << " (" << current_mesh.cave_anchor_points.size() << " anchors)" << std::endl;
                } else if (event.key.key == SDLK_F6) {
                    render_options.show_goldberg_grid = !render_options.show_goldberg_grid;
                    std::cout << "Goldberg grid " << (render_options.show_goldberg_grid ? "shown" : "hidden") << std::endl;
                } else if (event.key.key == SDLK_F7) {
                    render_options.show_mesh_wire = !render_options.show_mesh_wire;
                    std::cout << "Mesh wire " << (render_options.show_mesh_wire ? "shown" : "hidden") << std::endl;
                } else if (event.key.key == SDLK_F8) {
                    render_options.show_points = !render_options.show_points;
                    std::cout << "Point samples " << (render_options.show_points ? "shown" : "hidden") << std::endl;
                } else if (event.key.key == SDLK_F10) {
                    render_options.show_voxels = !render_options.show_voxels;
                    std::cout << "Voxels " << (render_options.show_voxels ? "shown" : "hidden") << std::endl;
                } else if (event.key.key == SDLK_F12) {
                    render_options.show_surface_net = !render_options.show_surface_net;
                    std::cout << "Surface nets " << (render_options.show_surface_net ? "shown" : "hidden") << std::endl;
                } else if (event.key.key == SDLK_F11) {
                    mesh_config.svo_depth = DefaultSvoDepth;
                    mesh_config.surface_net_depth = DefaultSvoDepth;
                    mesh_config.local_surface_net_depth = mesh_config.local_surface_net_depth >= HighSvoDepth ? DefaultSvoDepth : HighSvoDepth;
                    mesh_config.voxel_features.cave_depth = mesh_config.local_surface_net_depth;
                    mesh_config.svo_debug_draw_depth = SvoDebugDrawDepth;
                    request_voxel_mesh_rebuild("cave depth toggle");
                    std::cout << "Cave SVO detail depth " << mesh_config.voxel_features.cave_depth
                              << " rebuild requested (global exterior replacement disabled; SVO depth " << mesh_config.svo_depth
                              << ")" << std::endl;
                } else if (event.key.key == SDLK_F9) {
                    render_options.follow_ship = !render_options.follow_ship;
                    if (render_options.follow_ship != relative_mouse_enabled) {
                        if (!render_options.follow_ship) {
                            int width = 960;
                            int height = 540;
                            SDL_GetWindowSizeInPixels(window, &width, &height);
                            SDL_WarpMouseInWindow(window, static_cast<float>(width) * 0.5f, static_cast<float>(height) * 0.5f);
                        }
                        if (SDL_SetWindowRelativeMouseMode(window, render_options.follow_ship)) {
                            relative_mouse_enabled = render_options.follow_ship;
                        } else {
                            std::cerr << "SDL_SetWindowRelativeMouseMode failed: " << SDL_GetError() << std::endl;
                        }
                    }
                    std::cout << "Ship follow camera " << (render_options.follow_ship ? "enabled" : "disabled") << std::endl;
                } else if (event.key.key == SDLK_F4) {
                    mesh_config.enable_fractures = !mesh_config.enable_fractures;
                    request_full_mesh_rebuild("fracture toggle");
                    std::cout << "Fractures " << (mesh_config.enable_fractures ? "enabled" : "disabled") << std::endl;
                } else if (event.key.key == SDLK_F5) {
                    ++mesh_config.fracture_seed;
                    mesh_config.enable_fractures = true;
                    request_full_mesh_rebuild("fracture seed");
                    std::cout << "Fracture seed " << mesh_config.fracture_seed << std::endl;
                } else if (event.key.key == SDLK_F13) {
                    voxel_edits.digs.clear();
                    mesh_config.voxel_edits = voxel_edits;
                    request_voxel_mesh_rebuild("clear cave edits");
                    std::cout << "Cleared dig edits; rebuild requested" << std::endl;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT) {
                const ae::CameraView dig_view = render_options.follow_ship ? make_ship_follow_view(spaceship) : make_free_camera_view(camera);
                const ae::Vec3 dig_direction = ae::normalize(dig_view.target - dig_view.eye);
                ae::Vec3 hit_mesh = {};
                const bool hit_surface_net = ray_surface_net_intersection(
                    current_mesh.surface_net,
                    dig_view.eye / ae::PlanetRadiusKilometers,
                    dig_direction,
                    hit_mesh
                );
                const bool hit_cave_feature = hit_surface_net || ray_cave_feature_intersection(
                    current_mesh.voxel_features,
                    dig_view.eye / ae::PlanetRadiusKilometers,
                    dig_direction,
                    hit_mesh
                );
                if (hit_cave_feature) {
                    voxel_edits.digs.push_back({
                        hit_mesh,
                        8.0f,
                        voxel_edits.local_depth,
                    });
                    mesh_config.voxel_edits = voxel_edits;
                    mesh_config.lod_camera_position = dig_view.eye;
                    request_voxel_mesh_rebuild("cave dig");
                    std::cout << "Dig edit " << voxel_edits.digs.size()
                              << ": radius 8 km, local depth " << voxel_edits.local_depth
                              << " cave voxel rebuild requested" << std::endl;
                }
            } else if (event.type == SDL_EVENT_MOUSE_MOTION && render_options.follow_ship) {
                ship_mouse_yaw += static_cast<float>(event.motion.xrel);
                ship_mouse_pitch -= static_cast<float>(event.motion.yrel);
            } else if (event.type == SDL_EVENT_MOUSE_MOTION && (event.motion.state & SDL_BUTTON_RMASK) != 0) {
                camera.yaw += static_cast<float>(event.motion.xrel) * 0.0035f;
                camera.pitch = std::clamp(camera.pitch - static_cast<float>(event.motion.yrel) * 0.0035f, -1.48f, 1.48f);
                mesh_config.lod_camera_position = free_camera_eye(camera);
                const ae::Vec3 new_lod_focus = ae::normalize(mesh_config.lod_camera_position);
                if (ae::length(new_lod_focus - requested_mesh_lod_focus) >= LodRebuildDistance) {
                    request_full_mesh_rebuild("orbit camera");
                }
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                camera.distance = std::clamp(
                    camera.distance * std::pow(0.88f, event.wheel.y),
                    ae::PlanetRadiusKilometers * 1.02f,
                    ae::PlanetRadiusKilometers * 12.0f
                );
            }
        }

        const uint64_t update_time = SDL_GetTicksNS();
        const float dt = static_cast<float>(update_time - last_update_time) / 1'000'000'000.0f;
        last_update_time = update_time;
        const float safe_input_dt = std::max(dt, 0.0001f);
        ae::SpaceshipInput ship_input;
        ship_input.yaw = (ship_mouse_yaw / safe_input_dt) * 0.00012f;
        ship_input.pitch = (ship_mouse_pitch / safe_input_dt) * 0.00012f;
        int key_count = 0;
        const bool* keys = SDL_GetKeyboardState(&key_count);
        if (keys != nullptr) {
            auto key_down = [&](SDL_Scancode scancode) {
                return static_cast<int>(scancode) < key_count && keys[scancode];
            };
            ship_input.throttle += key_down(SDL_SCANCODE_W) ? 1.0f : 0.0f;
            ship_input.throttle -= key_down(SDL_SCANCODE_S) ? 1.0f : 0.0f;
            ship_input.roll -= key_down(SDL_SCANCODE_A) ? 1.0f : 0.0f;
            ship_input.roll += key_down(SDL_SCANCODE_D) ? 1.0f : 0.0f;
            if (!render_options.follow_ship) {
                const ae::Vec3 forward = free_camera_forward(camera);
                const ae::Vec3 right = free_camera_right(camera);
                ae::Vec3 movement = {};
                movement = movement + forward * (key_down(SDL_SCANCODE_I) ? 1.0f : 0.0f);
                movement = movement - forward * (key_down(SDL_SCANCODE_K) ? 1.0f : 0.0f);
                movement = movement - right * (key_down(SDL_SCANCODE_J) ? 1.0f : 0.0f);
                movement = movement + right * (key_down(SDL_SCANCODE_L) ? 1.0f : 0.0f);
                movement.y += key_down(SDL_SCANCODE_E) ? 1.0f : 0.0f;
                movement.y -= key_down(SDL_SCANCODE_Q) ? 1.0f : 0.0f;
                if (ae::length(movement) > 0.000001f) {
                    camera.target = camera.target + ae::normalize(movement) * (camera.move_speed * dt);
                    mesh_config.lod_camera_position = free_camera_eye(camera);
                    const ae::Vec3 new_lod_focus = ae::normalize(mesh_config.lod_camera_position);
                    if (ae::length(new_lod_focus - requested_mesh_lod_focus) >= LodRebuildDistance) {
                        request_full_mesh_rebuild("free camera move");
                    }
                }
            }
        }
        ae::update_spaceship(spaceship, ship_input, dt);
        if (benchmark_mode && !render_options.follow_ship) {
            benchmark_time += static_cast<double>(dt);
            camera.target = {};
            camera.yaw = static_cast<float>(benchmark_time * 0.18);
            camera.pitch = -0.30f + std::sin(static_cast<float>(benchmark_time) * 0.37f) * 0.20f;
            camera.distance = ae::PlanetRadiusKilometers * (1.08f + 0.16f * (0.5f + 0.5f * std::sin(static_cast<float>(benchmark_time) * 0.21f)));
        }
        const ae::CameraView camera_view = render_options.follow_ship ? make_ship_follow_view(spaceship) : make_free_camera_view(camera);
        mesh_config.lod_camera_position = camera_view.eye;
        const ae::Vec3 frame_lod_focus = ae::normalize(mesh_config.lod_camera_position);
        if (ae::length(frame_lod_focus - requested_mesh_lod_focus) >= LodRebuildDistance) {
            request_full_mesh_rebuild(render_options.follow_ship ? "ship camera LOD" : "camera LOD");
        }

        if (mesh_build_pending && pending_mesh_build.wait_for(NoWait) == std::future_status::ready) {
            MeshBuildResult result = pending_mesh_build.get();
            mesh_build_pending = false;
            if (result.revision < requested_mesh_revision) {
                std::cout << "Discarded stale mesh build revision " << result.revision
                          << " (latest requested " << requested_mesh_revision << ")" << std::endl;
                if (!mesh_rebuild_requested) {
                    build_progress->set(0.0f, "Idle");
                }
            } else if (result.validation.ok) {
                current_mesh = result.mesh;
                renderer.update_mesh(result.mesh);
                last_mesh_lod_focus = result.lod_focus;
                std::cout << (result.mode == MeshBuildMode::VoxelsOnly ? "Voxel rebuild OK: " : "Mesh rebuild OK: ")
                          << result.validation.message << std::endl;
                const ae::RendererPerfStats& renderer_stats = renderer.perf_stats();
                std::cout << "Perf [" << result.reason << "]: queue " << result.queue_wait_ms
                          << " ms, async " << result.async_build_ms
                          << " ms, validation " << result.validation_ms
                          << " ms, upload " << renderer_stats.upload_ms
                          << " ms, mesh upload " << (renderer_stats.mesh_upload_bytes / (1024.0 * 1024.0))
                          << " MiB, surface-net upload " << (renderer_stats.surface_net_upload_bytes / (1024.0 * 1024.0))
                          << " MiB, cave anchors " << result.mesh.cave_anchor_points.size() << std::endl;
                if (!result.mesh.perf.cave_features.empty()) {
                    for (const ae::CaveFeatureBuildStats& cave_stats : result.mesh.perf.cave_features) {
                        std::cout << "  Cave " << cave_stats.feature_id
                                  << " cell " << cave_stats.owner_cell_id
                                  << ": depth " << cave_stats.depth
                                  << ", " << cave_stats.vertices << " vertices, "
                                  << cave_stats.triangles << " triangles, "
                                  << cave_stats.occupied_voxels << " occupied voxels, "
                                  << cave_stats.candidate_cubes << " candidate cubes, "
                                  << cave_stats.surface_net_ms << " ms"
                                  << " [occ " << cave_stats.occupancy_ms
                                  << ", candidates " << cave_stats.candidate_ms
                                  << ", compact " << cave_stats.compact_ms
                                  << ", verts " << cave_stats.vertex_ms
                                  << ", quads " << cave_stats.quad_ms
                                  << "]" << std::endl;
                    }
                }
            } else {
                std::cerr << result.validation.message << std::endl;
                last_mesh_lod_focus = result.lod_focus;
                build_progress->set(0.0f, "Build failed");
            }
        }

        if (!mesh_build_pending && mesh_rebuild_requested) {
            ae::MarchingCubesConfig async_config = mesh_config;
            async_config.lod_camera_position = requested_mesh_camera_position;
            const uint32_t async_revision = requested_mesh_revision;
            const MeshBuildMode async_mode = requested_mesh_build_mode;
            const std::string async_reason = requested_mesh_build_reason;
            const auto async_queued_at = requested_mesh_queued_at;
            mesh_rebuild_requested = false;
            requested_mesh_build_mode = MeshBuildMode::Full;
            if (async_mode == MeshBuildMode::VoxelsOnly) {
                ae::QuantizedMesh voxel_base_mesh = make_voxel_rebuild_source(current_mesh);
                build_progress->set(0.0f, async_config.local_surface_net_depth >= HighSvoDepth ? "Queued depth-16 detail rebuild" : "Queued voxel rebuild");
                pending_mesh_build = std::async(
                    std::launch::async,
                    build_voxel_mesh_async,
                    std::move(voxel_base_mesh),
                    async_config,
                    async_revision,
                    async_reason,
                    async_queued_at,
                    build_progress
                );
            } else {
                build_progress->set(0.0f, async_config.local_surface_net_depth >= HighSvoDepth ? "Queued depth-16 detail rebuild" : "Queued mesh rebuild");
                pending_mesh_build = std::async(
                    std::launch::async,
                    build_lod_mesh_async,
                    std::cref(topology),
                    std::cref(points),
                    async_config,
                    async_revision,
                    async_reason,
                    async_queued_at,
                    build_progress
                );
            }
            mesh_build_pending = true;
        }

        int width = 960;
        int height = 540;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        renderer.resize(width, height);
        renderer.render(camera_view, spaceship, render_options, show_fps, displayed_fps);
        if (mesh_build_pending) {
            renderer.render_progress_overlay(build_progress->progress());
        }
        SDL_GL_SwapWindow(window);
        const double frame_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frame_begin
        ).count();
        frame_times_ms[frame_time_cursor % frame_times_ms.size()] = frame_ms;
        ++frame_time_cursor;

        ++frames_since_fps_update;
        const uint64_t now = SDL_GetTicksNS();
        const uint64_t elapsed = now - fps_update_start;
        if (elapsed >= 250'000'000ull) {
            displayed_fps = static_cast<float>(frames_since_fps_update) * 1'000'000'000.0f / static_cast<float>(elapsed);
            frames_since_fps_update = 0;
            fps_update_start = now;
            recent_worst_frame_ms = 0.0;
            const uint32_t frame_sample_count = std::min<uint32_t>(frame_time_cursor, static_cast<uint32_t>(frame_times_ms.size()));
            for (uint32_t i = 0; i < frame_sample_count; ++i) {
                recent_worst_frame_ms = std::max(recent_worst_frame_ms, frame_times_ms[i]);
            }
            if (mesh_build_pending) {
                char percent_buffer[16] = {};
                std::snprintf(
                    percent_buffer,
                    sizeof(percent_buffer),
                    "%.6f%%",
                    std::clamp(build_progress->progress(), 0.0, 1.0) * 100.0
                );
                const std::string title = "AETHERONUS - " + build_progress->label_copy() + " (" + percent_buffer + ")";
                SDL_SetWindowTitle(window, title.c_str());
                window_title_shows_progress = true;
            } else if (show_fps) {
                const ae::RendererPerfStats& renderer_stats = renderer.perf_stats();
                char title_buffer[256] = {};
                std::snprintf(
                    title_buffer,
                    sizeof(title_buffer),
                    "AETHERONUS - FPS %.0f | frame %.2f ms worst %.2f | render %.2f ms | GPU mesh %.2f surf %.2f dbg %.2f | draws %u",
                    displayed_fps,
                    1000.0f / std::max(displayed_fps, 0.001f),
                    recent_worst_frame_ms,
                    renderer_stats.render_cpu_ms,
                    renderer_stats.gpu_mesh_ms,
                    renderer_stats.gpu_surface_net_ms,
                    renderer_stats.gpu_debug_ms,
                    renderer_stats.draw_calls
                );
                SDL_SetWindowTitle(window, title_buffer);
                window_title_shows_progress = true;
            } else if (window_title_shows_progress) {
                SDL_SetWindowTitle(window, "AETHERONUS");
                window_title_shows_progress = false;
            }
        }
    }

    if (relative_mouse_enabled) {
        SDL_SetWindowRelativeMouseMode(window, false);
    }
    renderer.shutdown();
    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
