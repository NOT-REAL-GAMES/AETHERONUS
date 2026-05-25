#include "aetheronus/debug_renderer.hpp"
#include "aetheronus/meshing.hpp"
#include "aetheronus/planet_scale.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/spaceship.hpp"
#include "aetheronus/topology.hpp"

#include <glad/glad.h>
#include <SDL3/SDL.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t DefaultSvoDepth = 8;
constexpr uint32_t ScreamSvoDepth = 13;
constexpr uint32_t SvoDebugDrawDepth = 8;

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

struct MeshBuildResult {
    ae::QuantizedMesh mesh;
    ae::QuantizedMeshValidation validation;
    ae::Vec3 camera_position;
    ae::Vec3 lod_focus;
    uint32_t revision = 0;
};

MeshBuildResult build_lod_mesh_async(
    const ae::GoldbergTopology& topology,
    const ae::PointCloud& points,
    ae::MarchingCubesConfig config,
    uint32_t revision
) {
    MeshBuildResult result;
    result.camera_position = config.lod_camera_position;
    result.lod_focus = ae::normalize(config.lod_camera_position);
    result.revision = revision;
    result.mesh = ae::build_quantized_marching_cubes(topology, points, config);
    result.validation = ae::validate_quantized_mesh(result.mesh);
    return result;
}

} // namespace

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_GL_ResetAttributes();
    if (!set_gl_attribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3, "major version") ||
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

    if (!GLAD_GL_VERSION_3_3) {
        std::cerr << "OpenGL 3.3 is required." << std::endl;
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
    mesh_config.svo_debug_draw_depth = SvoDebugDrawDepth;
    ae::VoxelEditSet voxel_edits;
    mesh_config.voxel_edits = voxel_edits;
    const ae::QuantizedMesh mesh = ae::build_quantized_marching_cubes(topology, points, mesh_config);
    const ae::TopologyValidation validation = ae::validate_topology(topology, static_cast<uint32_t>(points.size()));
    const ae::PointCloudValidation point_validation = ae::validate_point_cloud(topology, points);
    const ae::QuantizedMeshValidation mesh_validation = ae::validate_quantized_mesh(mesh);
    std::cout << validation.message << std::endl;
    std::cout << point_validation.message << std::endl;
    std::cout << mesh_validation.message << std::endl;
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

    ae::Vec3 last_mesh_lod_focus = ae::normalize(mesh_config.lod_camera_position);
    ae::Vec3 requested_mesh_lod_focus = last_mesh_lod_focus;
    ae::Vec3 requested_mesh_camera_position = mesh_config.lod_camera_position;
    uint32_t mesh_revision = 1;
    uint32_t requested_mesh_revision = mesh_revision;
    std::future<MeshBuildResult> pending_mesh_build;
    bool mesh_build_pending = false;
    bool mesh_rebuild_requested = false;
    bool show_fps = false;
    ae::DebugRenderOptions render_options;
    ae::SpaceshipState spaceship;
    bool relative_mouse_enabled = false;
    float displayed_fps = 0.0f;
    uint32_t frames_since_fps_update = 0;
    uint64_t fps_update_start = SDL_GetTicksNS();
    uint64_t last_update_time = fps_update_start;
    constexpr float LodRebuildDistance = 0.12f;
    constexpr std::chrono::milliseconds NoWait{0};
    auto request_mesh_rebuild = [&]() {
        requested_mesh_revision = ++mesh_revision;
        requested_mesh_lod_focus = ae::normalize(mesh_config.lod_camera_position);
        requested_mesh_camera_position = mesh_config.lod_camera_position;
        mesh_rebuild_requested = true;
    };

    bool running = true;
    while (running) {
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
                        requested_mesh_lod_focus = new_lod_focus;
                        requested_mesh_camera_position = mesh_config.lod_camera_position;
                        requested_mesh_revision = ++mesh_revision;
                        mesh_rebuild_requested = true;
                    }
                } else if (event.key.key == SDLK_F3) {
                    show_fps = !show_fps;
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
                    const bool enable_scream_mode = mesh_config.svo_depth < ScreamSvoDepth;
                    mesh_config.svo_depth = enable_scream_mode ? ScreamSvoDepth : DefaultSvoDepth;
                    mesh_config.svo_debug_draw_depth = SvoDebugDrawDepth;
                    request_mesh_rebuild();
                    std::cout << "SVO depth " << mesh_config.svo_depth
                              << " rebuild requested (debug draw depth " << mesh_config.svo_debug_draw_depth
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
                    request_mesh_rebuild();
                    std::cout << "Fractures " << (mesh_config.enable_fractures ? "enabled" : "disabled") << std::endl;
                } else if (event.key.key == SDLK_F5) {
                    ++mesh_config.fracture_seed;
                    mesh_config.enable_fractures = true;
                    request_mesh_rebuild();
                    std::cout << "Fracture seed " << mesh_config.fracture_seed << std::endl;
                } else if (event.key.key == SDLK_F13) {
                    voxel_edits.digs.clear();
                    mesh_config.voxel_edits = voxel_edits;
                    request_mesh_rebuild();
                    std::cout << "Cleared dig edits; rebuild requested" << std::endl;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT) {
                const ae::CameraView dig_view = render_options.follow_ship ? make_ship_follow_view(spaceship) : make_free_camera_view(camera);
                ae::Vec3 hit_km = {};
                if (ray_sphere_intersection(dig_view.eye, dig_view.target - dig_view.eye, ae::PlanetRadiusKilometers, hit_km)) {
                    voxel_edits.digs.push_back({
                        hit_km / ae::PlanetRadiusKilometers,
                        8.0f,
                        voxel_edits.local_depth,
                    });
                    mesh_config.voxel_edits = voxel_edits;
                    mesh_config.lod_camera_position = dig_view.eye;
                    request_mesh_rebuild();
                    std::cout << "Dig edit " << voxel_edits.digs.size()
                              << ": radius 8 km, local depth " << voxel_edits.local_depth
                              << " rebuild requested" << std::endl;
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
                    requested_mesh_lod_focus = new_lod_focus;
                    requested_mesh_camera_position = mesh_config.lod_camera_position;
                    requested_mesh_revision = ++mesh_revision;
                    mesh_rebuild_requested = true;
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
                        requested_mesh_lod_focus = new_lod_focus;
                        requested_mesh_camera_position = mesh_config.lod_camera_position;
                        requested_mesh_revision = ++mesh_revision;
                        mesh_rebuild_requested = true;
                    }
                }
            }
        }
        ae::update_spaceship(spaceship, ship_input, dt);
        const ae::CameraView camera_view = render_options.follow_ship ? make_ship_follow_view(spaceship) : make_free_camera_view(camera);
        mesh_config.lod_camera_position = camera_view.eye;
        const ae::Vec3 frame_lod_focus = ae::normalize(mesh_config.lod_camera_position);
        if (ae::length(frame_lod_focus - requested_mesh_lod_focus) >= LodRebuildDistance) {
            requested_mesh_lod_focus = frame_lod_focus;
            requested_mesh_camera_position = mesh_config.lod_camera_position;
            requested_mesh_revision = ++mesh_revision;
            mesh_rebuild_requested = true;
        }

        if (mesh_build_pending && pending_mesh_build.wait_for(NoWait) == std::future_status::ready) {
            MeshBuildResult result = pending_mesh_build.get();
            mesh_build_pending = false;
            if (result.validation.ok) {
                renderer.update_mesh(result.mesh);
                last_mesh_lod_focus = result.lod_focus;
                std::cout << "Mesh rebuild OK: " << result.validation.message << std::endl;
            } else {
                std::cerr << result.validation.message << std::endl;
                last_mesh_lod_focus = result.lod_focus;
            }
        }

        if (!mesh_build_pending && mesh_rebuild_requested) {
            ae::MarchingCubesConfig async_config = mesh_config;
            async_config.lod_camera_position = requested_mesh_camera_position;
            const uint32_t async_revision = requested_mesh_revision;
            mesh_rebuild_requested = false;
            pending_mesh_build = std::async(
                std::launch::async,
                build_lod_mesh_async,
                std::cref(topology),
                std::cref(points),
                async_config,
                async_revision
            );
            mesh_build_pending = true;
        }

        int width = 960;
        int height = 540;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        renderer.resize(width, height);
        renderer.render(camera_view, spaceship, render_options, show_fps, displayed_fps);
        SDL_GL_SwapWindow(window);

        ++frames_since_fps_update;
        const uint64_t now = SDL_GetTicksNS();
        const uint64_t elapsed = now - fps_update_start;
        if (elapsed >= 250'000'000ull) {
            displayed_fps = static_cast<float>(frames_since_fps_update) * 1'000'000'000.0f / static_cast<float>(elapsed);
            frames_since_fps_update = 0;
            fps_update_start = now;
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
