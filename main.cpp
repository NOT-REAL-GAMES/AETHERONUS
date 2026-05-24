#include "aetheronus/debug_renderer.hpp"
#include "aetheronus/meshing.hpp"
#include "aetheronus/point_cloud.hpp"
#include "aetheronus/topology.hpp"

#include <glad/glad.h>
#include <SDL3/SDL.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

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

    const ae::GoldbergTopology topology = ae::build_goldberg_topology(2);
    const std::vector<ae::PointSample> points = ae::build_surface_point_cloud(topology);
    const ae::QuantizedMesh mesh = ae::build_quantized_marching_cubes(topology, points);
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

    ae::OrbitCamera camera;
    bool show_fps = false;
    float displayed_fps = 0.0f;
    uint32_t frames_since_fps_update = 0;
    uint64_t fps_update_start = SDL_GetTicksNS();

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                if (event.key.key == SDLK_ESCAPE) {
                    running = false;
                } else if (event.key.key == SDLK_R) {
                    camera = ae::OrbitCamera{};
                } else if (event.key.key == SDLK_F3) {
                    show_fps = !show_fps;
                }
            } else if (event.type == SDL_EVENT_MOUSE_MOTION && (event.motion.state & SDL_BUTTON_LMASK) != 0) {
                camera.yaw += event.motion.xrel * 0.008f;
                camera.pitch = std::clamp(camera.pitch + event.motion.yrel * 0.008f, -1.35f, 1.35f);
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                camera.distance = std::clamp(camera.distance - event.wheel.y * 0.22f, 1.6f, 8.0f);
            }
        }

        int width = 960;
        int height = 540;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        renderer.resize(width, height);
        renderer.render(camera, show_fps, displayed_fps);
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

    renderer.shutdown();
    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
