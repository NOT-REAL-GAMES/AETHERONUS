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
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t DefaultSvoDepth = 8;
constexpr uint32_t HighSvoDepth = 16;
constexpr uint32_t StreamedCaveDepth = 4;
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
    ae::Vec3& hit_mesh,
    float* hit_t = nullptr
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
    if (hit_t != nullptr) {
        *hit_t = closest_t;
    }
    return true;
}

bool ray_quantized_mesh_intersection(
    const ae::QuantizedMesh& mesh,
    ae::Vec3 origin_mesh,
    ae::Vec3 direction,
    ae::Vec3& hit_mesh
) {
    if (mesh.vertices.empty()) {
        return false;
    }

    direction = ae::normalize(direction);
    bool found = false;
    float closest_t = std::numeric_limits<float>::max();
    auto scan_indices = [&](const std::vector<uint32_t>& indices) {
        for (uint32_t i = 0; i + 2u < indices.size(); i += 3u) {
            const uint32_t ia = indices[i];
            const uint32_t ib = indices[i + 1u];
            const uint32_t ic = indices[i + 2u];
            if (ia >= mesh.vertices.size() || ib >= mesh.vertices.size() || ic >= mesh.vertices.size()) {
                continue;
            }

            float t = 0.0f;
            if (ray_triangle_intersection(
                    origin_mesh,
                    direction,
                    mesh.vertices[ia].position,
                    mesh.vertices[ib].position,
                    mesh.vertices[ic].position,
                    t
                ) && t < closest_t) {
                closest_t = t;
                found = true;
            }
        }
    };

    scan_indices(mesh.triangle_indices);
    scan_indices(mesh.stitch_triangle_indices);
    if (!found) {
        return false;
    }

    hit_mesh = origin_mesh + direction * closest_t;
    return true;
}

struct CaveFeatureRayHit {
    uint32_t feature_id = 0;
    ae::Vec3 hit_mesh = {};
};

constexpr float CaveViewTargetMaxDistanceKm = 900.0f;
constexpr float CaveClickTargetMaxDistanceKm = 1200.0f;
constexpr uint32_t CaveRayCandidateLimit = 32u;

bool ray_cave_feature_hit(
    const std::vector<ae::LocalVoxelFeature>& features,
    ae::Vec3 origin_mesh,
    ae::Vec3 direction,
    CaveFeatureRayHit& hit,
    float max_distance_km = CaveViewTargetMaxDistanceKm
) {
    direction = ae::normalize(direction);
    const float max_distance_mesh = ae::kilometers_to_world_units(max_distance_km);
    const float max_distance_sq = max_distance_mesh * max_distance_mesh;
    struct Candidate {
        float projected_distance = 0.0f;
        const ae::LocalVoxelFeature* feature = nullptr;
    };
    std::array<Candidate, CaveRayCandidateLimit> candidates = {};
    uint32_t candidate_count = 0u;

    auto keep_candidate = [&](float projected_distance, const ae::LocalVoxelFeature& feature) {
        if (candidate_count < CaveRayCandidateLimit) {
            candidates[candidate_count++] = {projected_distance, &feature};
            return;
        }
        uint32_t worst_index = 0u;
        float worst_distance = candidates[0].projected_distance;
        for (uint32_t i = 1u; i < CaveRayCandidateLimit; ++i) {
            if (candidates[i].projected_distance > worst_distance) {
                worst_distance = candidates[i].projected_distance;
                worst_index = i;
            }
        }
        if (projected_distance < worst_distance) {
            candidates[worst_index] = {projected_distance, &feature};
        }
    };

    for (const ae::LocalVoxelFeature& feature : features) {
        if (feature.kind != ae::VoxelFeatureKind::CaveSystem) {
            continue;
        }

        const ae::Vec3 to_center = feature.center_mesh - origin_mesh;
        const float center_distance_sq = ae::dot(to_center, to_center);
        if (center_distance_sq > max_distance_sq) {
            continue;
        }

        const float projected_distance = ae::dot(to_center, direction);
        if (projected_distance <= 0.0f || projected_distance > max_distance_mesh) {
            continue;
        }

        const float entrance_radius = ae::kilometers_to_world_units(feature.entrance_radius_km) * 1.65f;
        const float perpendicular_sq = center_distance_sq - projected_distance * projected_distance;
        if (perpendicular_sq > entrance_radius * entrance_radius) {
            continue;
        }

        keep_candidate(projected_distance, feature);
    }

    bool found = false;
    float closest_t = std::numeric_limits<float>::max();
    ae::Vec3 closest_hit = {};
    uint32_t closest_feature_id = 0;

    for (uint32_t candidate_index = 0u; candidate_index < candidate_count; ++candidate_index) {
        const ae::LocalVoxelFeature* feature = candidates[candidate_index].feature;
        if (feature == nullptr) {
            continue;
        }

        const float denominator = ae::dot(direction, feature->normal_mesh);
        if (std::fabs(denominator) <= 0.000001f) {
            continue;
        }

        const float t = ae::dot(feature->center_mesh - origin_mesh, feature->normal_mesh) / denominator;
        if (t <= 0.0f || t >= closest_t || t > max_distance_mesh) {
            continue;
        }

        const ae::Vec3 plane_hit = origin_mesh + direction * t;
        const ae::Vec3 offset = plane_hit - feature->center_mesh;
        const float u = ae::dot(offset, feature->tangent_mesh);
        const float v = ae::dot(offset, feature->bitangent_mesh);
        const float entrance_radius = ae::kilometers_to_world_units(feature->entrance_radius_km) * 1.15f;
        if ((u * u + v * v) > entrance_radius * entrance_radius) {
            continue;
        }

        const float seed_depth = std::max(
            ae::kilometers_to_world_units(2.0f),
            ae::kilometers_to_world_units(feature->tunnel_radius_km) * 0.45f
        );
        closest_hit = plane_hit - feature->normal_mesh * seed_depth;
        closest_t = t;
        closest_feature_id = feature->feature_id;
        found = true;
    }

    if (!found) {
        return false;
    }

    hit.feature_id = closest_feature_id;
    hit.hit_mesh = closest_hit;
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

float mesh_bounds_radius(const ae::QuantizedMesh& mesh) {
    float radius = 1.0f;
    for (const ae::QuantizedMeshVertex& vertex : mesh.vertices) {
        radius = std::max(radius, ae::length(vertex.position));
    }
    return radius * 1.025f;
}

void append_surface_mesh(ae::SurfaceNetMesh& destination, const ae::SurfaceNetMesh& source) {
    const uint32_t vertex_offset = static_cast<uint32_t>(destination.vertices.size());
    destination.vertices.insert(destination.vertices.end(), source.vertices.begin(), source.vertices.end());
    destination.normals.insert(destination.normals.end(), source.normals.begin(), source.normals.end());
    destination.vertex_depths.insert(destination.vertex_depths.end(), source.vertex_depths.begin(), source.vertex_depths.end());
    destination.triangle_indices.reserve(destination.triangle_indices.size() + source.triangle_indices.size());
    for (uint32_t index : source.triangle_indices) {
        destination.triangle_indices.push_back(vertex_offset + index);
    }
}

constexpr uint16_t TerrainHeightHole = 0u;
constexpr uint16_t TerrainHeightGoldberg = 8191u;
constexpr uint16_t TerrainHeightMax = 16383u;
constexpr uint32_t TerrainMaskResolution = 128u;
constexpr float TerrainMaskRadiusKm = 64.0f;

ae::TerrainHeightMask make_terrain_height_mask(ae::Vec3 center_mesh) {
    const ae::Vec3 normal = ae::normalize(center_mesh);
    const ae::Vec3 reference = std::fabs(normal.y) < 0.92f ? ae::Vec3{0.0f, 1.0f, 0.0f} : ae::Vec3{1.0f, 0.0f, 0.0f};
    const ae::Vec3 tangent = ae::normalize(ae::cross(reference, normal));
    const ae::Vec3 bitangent = ae::cross(normal, tangent);

    ae::TerrainHeightMask mask;
    mask.center_mesh = normal;
    mask.tangent_mesh = tangent;
    mask.bitangent_mesh = bitangent;
    mask.radius_km = TerrainMaskRadiusKm;
    mask.resolution = TerrainMaskResolution;
    mask.heights.assign(static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution), TerrainHeightGoldberg);
    mask.revision = 1u;
    return mask;
}

bool terrain_mask_project(const ae::TerrainHeightMask& mask, ae::Vec3 point_mesh, float& px, float& py) {
    if (mask.resolution == 0u || mask.radius_km <= 0.0f || ae::length(mask.center_mesh) <= 0.000001f) {
        return false;
    }
    const float radius_mesh = ae::kilometers_to_world_units(mask.radius_km);
    const ae::Vec3 relative = ae::normalize(point_mesh) - ae::normalize(mask.center_mesh);
    const float local_x = ae::dot(relative, mask.tangent_mesh);
    const float local_y = ae::dot(relative, mask.bitangent_mesh);
    const float u = local_x / (radius_mesh * 2.0f) + 0.5f;
    const float v = local_y / (radius_mesh * 2.0f) + 0.5f;
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return false;
    }
    px = u * static_cast<float>(mask.resolution - 1u);
    py = v * static_cast<float>(mask.resolution - 1u);
    return true;
}

ae::Vec3 terrain_mask_pixel_to_mesh(const ae::TerrainHeightMask& mask, uint32_t x, uint32_t y) {
    const float resolution_minus_one = static_cast<float>(std::max(1u, mask.resolution - 1u));
    const float radius_mesh = ae::kilometers_to_world_units(mask.radius_km);
    const float local_x = ((static_cast<float>(x) / resolution_minus_one) - 0.5f) * radius_mesh * 2.0f;
    const float local_y = ((static_cast<float>(y) / resolution_minus_one) - 0.5f) * radius_mesh * 2.0f;
    const float surface_radius = std::max(0.000001f, ae::length(mask.center_mesh));
    return ae::normalize(mask.center_mesh + mask.tangent_mesh * local_x + mask.bitangent_mesh * local_y) * surface_radius;
}

size_t terrain_mask_for_point(ae::VoxelEditSet& edits, ae::Vec3 point_mesh) {
    size_t best = edits.terrain_masks.size();
    float best_margin = -1.0f;
    for (size_t i = 0; i < edits.terrain_masks.size(); ++i) {
        float px = 0.0f;
        float py = 0.0f;
        ae::TerrainHeightMask& mask = edits.terrain_masks[i];
        if (!terrain_mask_project(mask, point_mesh, px, py)) {
            continue;
        }
        const float margin = std::min({px, py, static_cast<float>(mask.resolution - 1u) - px, static_cast<float>(mask.resolution - 1u) - py});
        if (margin > best_margin) {
            best_margin = margin;
            best = i;
        }
    }
    if (best != edits.terrain_masks.size() && best_margin >= static_cast<float>(TerrainMaskResolution) * 0.12f) {
        return best;
    }

    edits.terrain_masks.push_back(make_terrain_height_mask(point_mesh));
    return edits.terrain_masks.size() - 1u;
}

void paint_terrain_mask_line(
    ae::TerrainHeightMask& mask,
    ae::Vec3 from_mesh,
    ae::Vec3 to_mesh,
    float brush_radius_km,
    uint16_t height_value
) {
    if (mask.heights.size() != static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution)) {
        mask.heights.assign(static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution), TerrainHeightGoldberg);
    }
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    if (!terrain_mask_project(mask, from_mesh, x0, y0) || !terrain_mask_project(mask, to_mesh, x1, y1)) {
        return;
    }

    const float brush_pixels = std::max(
        1.0f,
        (brush_radius_km / std::max(1.0f, mask.radius_km * 2.0f)) * static_cast<float>(mask.resolution)
    );
    const float min_xf = std::floor(std::min(x0, x1) - brush_pixels - 1.0f);
    const float max_xf = std::ceil(std::max(x0, x1) + brush_pixels + 1.0f);
    const float min_yf = std::floor(std::min(y0, y1) - brush_pixels - 1.0f);
    const float max_yf = std::ceil(std::max(y0, y1) + brush_pixels + 1.0f);
    const int min_x = std::clamp(static_cast<int>(min_xf), 0, static_cast<int>(mask.resolution - 1u));
    const int max_x = std::clamp(static_cast<int>(max_xf), 0, static_cast<int>(mask.resolution - 1u));
    const int min_y = std::clamp(static_cast<int>(min_yf), 0, static_cast<int>(mask.resolution - 1u));
    const int max_y = std::clamp(static_cast<int>(max_yf), 0, static_cast<int>(mask.resolution - 1u));

    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float segment_length_sq = dx * dx + dy * dy;
    const float brush_sq = brush_pixels * brush_pixels;
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const float px = static_cast<float>(x) + 0.5f;
            const float py = static_cast<float>(y) + 0.5f;
            const float t = segment_length_sq <= 0.000001f
                ? 0.0f
                : std::clamp(((px - x0) * dx + (py - y0) * dy) / segment_length_sq, 0.0f, 1.0f);
            const float cx = x0 + dx * t;
            const float cy = y0 + dy * t;
            const float dist_x = px - cx;
            const float dist_y = py - cy;
            if (dist_x * dist_x + dist_y * dist_y <= brush_sq) {
                mask.heights[static_cast<size_t>(y) * mask.resolution + static_cast<size_t>(x)] =
                    std::min<uint16_t>(mask.heights[static_cast<size_t>(y) * mask.resolution + static_cast<size_t>(x)], height_value);
            }
        }
    }
    ++mask.revision;
}

void paint_terrain_mask_point(ae::VoxelEditSet& edits, size_t mask_index, ae::Vec3 point_mesh, float brush_radius_km) {
    if (mask_index >= edits.terrain_masks.size()) {
        return;
    }
    paint_terrain_mask_line(edits.terrain_masks[mask_index], point_mesh, point_mesh, brush_radius_km, TerrainHeightHole);
}

std::vector<ae::LocalVoxelFeature> build_terrain_dig_holes(const ae::VoxelEditSet& edits) {
    std::vector<ae::LocalVoxelFeature> holes;
    constexpr uint32_t TerrainHoleFeatureIdBase = 0x80000000u;
    constexpr uint16_t TerrainHeightHoleValue = TerrainHeightHole;
    constexpr uint32_t TerrainHoleMinComponentPixels = 48u;
    constexpr float TerrainHoleMinCaveRadiusKm = 6.0f;

    for (uint32_t mask_index = 0; mask_index < edits.terrain_masks.size(); ++mask_index) {
        const ae::TerrainHeightMask& mask = edits.terrain_masks[mask_index];
        const size_t pixel_count = static_cast<size_t>(mask.resolution) * static_cast<size_t>(mask.resolution);
        if (mask.resolution == 0u || mask.heights.size() != pixel_count || ae::length(mask.center_mesh) <= 0.000001f) {
            continue;
        }

        std::vector<uint8_t> visited(pixel_count, 0u);
        std::vector<uint32_t> stack;
        std::vector<uint32_t> component;
        uint32_t component_index = 0u;
        stack.reserve(256u);
        component.reserve(256u);

        for (uint32_t start_y = 0; start_y < mask.resolution; ++start_y) {
            for (uint32_t start_x = 0; start_x < mask.resolution; ++start_x) {
                const uint32_t start_index = start_y * mask.resolution + start_x;
                if (visited[start_index] != 0u || mask.heights[start_index] != TerrainHeightHoleValue) {
                    continue;
                }

                stack.clear();
                component.clear();
                visited[start_index] = 1u;
                stack.push_back(start_index);
                while (!stack.empty()) {
                    const uint32_t current = stack.back();
                    stack.pop_back();
                    component.push_back(current);
                    const uint32_t cx = current % mask.resolution;
                    const uint32_t cy = current / mask.resolution;
                    for (int32_t oy = -1; oy <= 1; ++oy) {
                        for (int32_t ox = -1; ox <= 1; ++ox) {
                            if (ox == 0 && oy == 0) {
                                continue;
                            }
                            const int32_t nx = static_cast<int32_t>(cx) + ox;
                            const int32_t ny = static_cast<int32_t>(cy) + oy;
                            if (nx < 0 || ny < 0 || nx >= static_cast<int32_t>(mask.resolution) || ny >= static_cast<int32_t>(mask.resolution)) {
                                continue;
                            }
                            const uint32_t neighbor = static_cast<uint32_t>(ny) * mask.resolution + static_cast<uint32_t>(nx);
                            if (visited[neighbor] != 0u || mask.heights[neighbor] != TerrainHeightHoleValue) {
                                continue;
                            }
                            visited[neighbor] = 1u;
                            stack.push_back(neighbor);
                        }
                    }
                }

                if (component.size() < TerrainHoleMinComponentPixels) {
                    ++component_index;
                    continue;
                }

                ae::Vec3 center_sum{};
                for (uint32_t pixel : component) {
                    center_sum = center_sum + terrain_mask_pixel_to_mesh(mask, pixel % mask.resolution, pixel / mask.resolution);
                }
                if (component.empty() || ae::length(center_sum) <= 0.000001f) {
                    ++component_index;
                    continue;
                }

                const ae::Vec3 center = center_sum * (1.0f / static_cast<float>(component.size()));
                float radius_km = 0.0f;
                for (uint32_t pixel : component) {
                    const ae::Vec3 pixel_position = terrain_mask_pixel_to_mesh(mask, pixel % mask.resolution, pixel / mask.resolution);
                    radius_km = std::max(radius_km, ae::world_units_to_kilometers(ae::length(pixel_position - center)));
                }
                const float pixel_radius_km = mask.radius_km / static_cast<float>(std::max(1u, mask.resolution - 1u));
                radius_km = std::clamp(radius_km + pixel_radius_km * 2.0f, 4.0f, 56.0f);
                if (radius_km < TerrainHoleMinCaveRadiusKm) {
                    ++component_index;
                    continue;
                }

                const ae::Vec3 normal = ae::normalize(center);
                const ae::Vec3 tangent = ae::length(mask.tangent_mesh) > 0.000001f
                    ? ae::normalize(mask.tangent_mesh)
                    : ae::normalize(ae::cross(std::fabs(normal.y) < 0.92f ? ae::Vec3{0.0f, 1.0f, 0.0f} : ae::Vec3{1.0f, 0.0f, 0.0f}, normal));
                const ae::Vec3 bitangent = ae::normalize(ae::cross(normal, tangent));
                const uint32_t seed = 0x71d1d5u ^
                    (mask_index * 0x9e3779b9u) ^
                    (component_index * 0x85ebca6bu) ^
                    (mask.revision * 0xc2b2ae35u);

                ae::LocalVoxelFeature hole;
                hole.kind = ae::VoxelFeatureKind::CaveSystem;
                hole.feature_id = TerrainHoleFeatureIdBase | ((mask_index & 0x7fffu) << 16u) | (component_index & 0xffffu);
                hole.center_mesh = center;
                hole.normal_mesh = normal;
                hole.tangent_mesh = tangent;
                hole.bitangent_mesh = bitangent;
                hole.entrance_radius_km = radius_km;
                hole.tunnel_radius_km = std::max(3.0f, radius_km * 0.62f);
                hole.chamber_radius_km = std::max(8.0f, radius_km * 1.35f);
                hole.depth_km = std::clamp(radius_km * 3.5f, 18.0f, 140.0f);
                hole.svo_depth = edits.local_depth;
                hole.seed = seed;
                holes.push_back(hole);
                ++component_index;
            }
        }
    }
    return holes;
}

std::vector<ae::LocalVoxelFeature> merge_hole_lists(
    const std::vector<ae::LocalVoxelFeature>& cave_holes,
    const std::vector<ae::LocalVoxelFeature>& terrain_holes
) {
    std::vector<ae::LocalVoxelFeature> holes;
    holes.reserve(cave_holes.size() + terrain_holes.size());
    holes.insert(holes.end(), cave_holes.begin(), cave_holes.end());
    holes.insert(holes.end(), terrain_holes.begin(), terrain_holes.end());
    return holes;
}

bool is_terrain_hole_feature_id(uint32_t feature_id) {
    return (feature_id & 0x80000000u) != 0u;
}

uint32_t cave_priority_hash_u32(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

ae::Vec3 cave_feature_local_km(const ae::LocalVoxelFeature& feature, ae::Vec3 point_mesh) {
    const ae::Vec3 offset = point_mesh - feature.center_mesh;
    return {
        ae::world_units_to_kilometers(ae::dot(offset, feature.tangent_mesh)),
        ae::world_units_to_kilometers(ae::dot(offset, feature.bitangent_mesh)),
        ae::world_units_to_kilometers(-ae::dot(offset, feature.normal_mesh)),
    };
}

float cave_tunnel_sdf_km(ae::Vec3 local, float entrance_radius, float tunnel_radius, float depth) {
    const float z0 = 0.0f;
    const float z1 = depth;
    const float t = std::clamp((local.z - z0) / std::max(0.000001f, z1 - z0), 0.0f, 1.0f);
    const float radius = entrance_radius * (1.0f - t) + tunnel_radius * t;
    const float radial = std::sqrt(local.x * local.x + local.y * local.y) - radius;
    return std::max(radial, std::max(z0 - local.z, local.z - z1));
}

float cave_feature_sdf_km(const ae::LocalVoxelFeature& feature, ae::Vec3 point_mesh) {
    const ae::Vec3 local = cave_feature_local_km(feature, point_mesh);
    float sdf = cave_tunnel_sdf_km(
        local,
        feature.entrance_radius_km,
        feature.tunnel_radius_km,
        feature.depth_km
    );

    const uint32_t side_hash = cave_priority_hash_u32(feature.seed ^ 0x9e3779b9u);
    const float angle = (static_cast<float>(side_hash & 0xffffu) / 65535.0f) * 2.0f * ae::Pi;
    const float side_distance = feature.chamber_radius_km * 0.42f;
    const ae::Vec3 chamber_center{
        std::cos(angle) * side_distance,
        std::sin(angle) * side_distance,
        feature.depth_km * 0.86f,
    };
    sdf = std::min(sdf, ae::length(local - chamber_center) - feature.chamber_radius_km);

    const uint32_t lobe_hash = cave_priority_hash_u32(feature.seed ^ 0x51ed270bu);
    const float lobe_angle = (static_cast<float>(lobe_hash & 0xffffu) / 65535.0f) * 2.0f * ae::Pi;
    const ae::Vec3 lobe_center{
        std::cos(lobe_angle) * feature.chamber_radius_km * 0.72f,
        std::sin(lobe_angle) * feature.chamber_radius_km * 0.72f,
        feature.depth_km * 0.58f,
    };
    return std::min(sdf, ae::length(local - lobe_center) - feature.chamber_radius_km * 0.48f);
}

float cave_feature_stream_distance_km(const ae::LocalVoxelFeature& feature, ae::Vec3 point_mesh) {
    const float sdf = cave_feature_sdf_km(feature, point_mesh);
    if (sdf <= 0.0f) {
        return sdf;
    }
    const float center_distance = ae::world_units_to_kilometers(ae::length(point_mesh - feature.center_mesh));
    return std::min(sdf, center_distance);
}

uint32_t nearest_cave_feature_id_for_point(
    const std::vector<ae::LocalVoxelFeature>& features,
    ae::Vec3 point_mesh
) {
    uint32_t best_feature_id = 0u;
    float best_score = std::numeric_limits<float>::max();
    for (const ae::LocalVoxelFeature& feature : features) {
        if (feature.kind != ae::VoxelFeatureKind::CaveSystem) {
            continue;
        }
        const float score = std::fabs(cave_feature_sdf_km(feature, point_mesh));
        if (score < best_score) {
            best_score = score;
            best_feature_id = feature.feature_id;
        }
    }
    return best_feature_id;
}

std::vector<ae::LocalVoxelFeature> build_goldberg_clip_holes(
    const std::vector<ae::LocalVoxelFeature>& active_holes,
    const ae::VoxelEditSet& edits
) {
    std::vector<ae::LocalVoxelFeature> holes;
    (void)edits;
    holes.reserve(active_holes.size());
    for (const ae::LocalVoxelFeature& hole : active_holes) {
        if (hole.kind != ae::VoxelFeatureKind::CaveSystem ||
            hole.entrance_radius_km <= 0.0f ||
            ae::length(hole.center_mesh) <= 0.000001f) {
            continue;
        }
        holes.push_back(hole);
    }
    return holes;
}

struct CaveStreamStats {
    uint32_t descriptors = 0;
    uint32_t active = 0;
    uint32_t cached = 0;
    uint32_t queued = 0;
    uint32_t rendered = 0;
    double last_build_ms = 0.0;
};

struct CaveInteriorBuildResult {
    uint32_t feature_id = 0;
    uint32_t generation = 0;
    ae::SurfaceNetMesh surface;
    ae::CaveFeatureBuildStats stats;
    double build_ms = 0.0;
};

class CaveInteriorStreamer {
public:
    void configure(const ae::QuantizedMesh& mesh, const ae::MarchingCubesConfig& config) {
        base_features_ = mesh.voxel_features;
        config_ = config;
        config_.progress_callback = {};
        grid_radius_ = mesh_bounds_radius(mesh);
        cache_.clear();
        queued_ids_.clear();
        active_ids_.clear();
        active_holes_.clear();
        active_surface_ = {};
        active_surface_dirty_ = true;
        last_targeted_feature_id_ = 0;
        ++edit_generation_;
        last_rescore_ = std::chrono::steady_clock::time_point{};
        build_serial_ = 0;
        stats_ = {};
        refresh_feature_list(false);
        stats_.descriptors = static_cast<uint32_t>(features_.size());
        const uint32_t hardware_threads = std::max(1u, std::thread::hardware_concurrency());
        worker_count_ = std::max(1u, std::min(4u, hardware_threads > 1u ? hardware_threads - 1u : 1u));
    }

    void update_config(const ae::MarchingCubesConfig& config) {
        config_ = config;
        config_.progress_callback = {};
        refresh_feature_list(true);
    }

    bool update(const ae::CameraView& view, ae::Vec3 aim_direction, uint32_t targeted_feature_id) {
        current_eye_mesh_ = view.eye / ae::PlanetRadiusKilometers;
        bool changed = collect_ready_jobs();
        const auto now = std::chrono::steady_clock::now();
        const bool should_rescore =
            active_ids_.empty() ||
            targeted_feature_id != last_targeted_feature_id_ ||
            last_rescore_ == std::chrono::steady_clock::time_point{} ||
            now - last_rescore_ >= std::chrono::milliseconds(250);
        if (should_rescore) {
            last_targeted_feature_id_ = targeted_feature_id;
            last_rescore_ = now;
            changed = choose_active_features(view, aim_direction, targeted_feature_id) || changed;
        }
        queue_missing_active_features();
        evict_cache();
        if (active_surface_dirty_ || changed) {
            rebuild_active_surface();
            active_surface_dirty_ = false;
            changed = true;
        }
        stats_.descriptors = static_cast<uint32_t>(features_.size());
        stats_.active = static_cast<uint32_t>(active_ids_.size());
        stats_.cached = static_cast<uint32_t>(cache_.size());
        stats_.queued = static_cast<uint32_t>(jobs_.size());
        return changed;
    }

    void invalidate_feature(uint32_t feature_id) {
        if (feature_id == 0u) {
            return;
        }
        cache_.erase(feature_id);
        queued_ids_.erase(feature_id);
        ++edit_generation_;
        active_ids_.erase(std::remove(active_ids_.begin(), active_ids_.end(), feature_id), active_ids_.end());
        active_ids_.insert(active_ids_.begin(), feature_id);
        if (active_ids_.size() > ActiveBudget) {
            active_ids_.resize(ActiveBudget);
        }
        queue_feature(feature_id);
        active_surface_dirty_ = true;
    }

    void invalidate_active() {
        for (uint32_t feature_id : active_ids_) {
            cache_.erase(feature_id);
            queued_ids_.erase(feature_id);
        }
        ++edit_generation_;
        queue_missing_active_features();
        active_surface_dirty_ = true;
    }

    const ae::SurfaceNetMesh& active_surface() const {
        return active_surface_;
    }

    const std::vector<ae::LocalVoxelFeature>& features() const {
        return features_;
    }

    const std::vector<ae::LocalVoxelFeature>& active_holes() const {
        return active_holes_;
    }

    const CaveStreamStats& stats() const {
        return stats_;
    }

private:
    static constexpr uint32_t ActiveBudget = 48u;
    static constexpr uint32_t CacheBudget = 192u;
    static constexpr uint32_t RenderedInteriorBudget = 16u;
    static constexpr float AlwaysRenderedRadiusKm = 180.0f;

    struct CachedInterior {
        ae::SurfaceNetMesh surface;
        ae::CaveFeatureBuildStats stats;
        uint64_t last_used = 0;
        double build_ms = 0.0;
    };

    struct CaveBuildJob {
        uint32_t feature_id = 0;
        uint32_t generation = 0;
        std::future<CaveInteriorBuildResult> future;
    };

    void refresh_feature_list(bool activate_terrain_holes) {
        const std::vector<ae::LocalVoxelFeature> terrain_holes = build_terrain_dig_holes(config_.voxel_edits);
        std::unordered_set<uint32_t> old_terrain_ids = terrain_feature_ids_;
        std::unordered_set<uint32_t> new_terrain_ids;
        new_terrain_ids.reserve(terrain_holes.size());
        for (const ae::LocalVoxelFeature& feature : terrain_holes) {
            new_terrain_ids.insert(feature.feature_id);
        }

        bool terrain_features_changed = old_terrain_ids.size() != new_terrain_ids.size();
        if (!terrain_features_changed) {
            for (uint32_t feature_id : new_terrain_ids) {
                if (old_terrain_ids.find(feature_id) == old_terrain_ids.end()) {
                    terrain_features_changed = true;
                    break;
                }
            }
        }
        if (!terrain_features_changed && terrain_holes.size() == terrain_feature_count_) {
            for (const ae::LocalVoxelFeature& new_feature : terrain_holes) {
                auto old_it = std::find_if(features_.begin(), features_.end(), [&](const ae::LocalVoxelFeature& old_feature) {
                    return old_feature.feature_id == new_feature.feature_id;
                });
                if (old_it == features_.end() ||
                    old_it->seed != new_feature.seed ||
                    std::fabs(old_it->entrance_radius_km - new_feature.entrance_radius_km) > 0.001f ||
                    ae::length(old_it->center_mesh - new_feature.center_mesh) > 0.000001f) {
                    terrain_features_changed = true;
                    break;
                }
            }
        }

        features_ = merge_hole_lists(base_features_, terrain_holes);
        terrain_feature_ids_ = std::move(new_terrain_ids);
        terrain_feature_count_ = terrain_holes.size();
        stats_.descriptors = static_cast<uint32_t>(features_.size());

        if (!terrain_features_changed) {
            return;
        }

        for (uint32_t feature_id : old_terrain_ids) {
            cache_.erase(feature_id);
            queued_ids_.erase(feature_id);
        }
        for (uint32_t feature_id : terrain_feature_ids_) {
            cache_.erase(feature_id);
            queued_ids_.erase(feature_id);
        }
        active_ids_.erase(std::remove_if(active_ids_.begin(), active_ids_.end(), is_terrain_hole_feature_id), active_ids_.end());
        if (activate_terrain_holes) {
            for (auto it = terrain_holes.rbegin(); it != terrain_holes.rend(); ++it) {
                active_ids_.insert(active_ids_.begin(), it->feature_id);
            }
            if (active_ids_.size() > ActiveBudget) {
                active_ids_.resize(ActiveBudget);
            }
        }
        ++edit_generation_;
        last_rescore_ = std::chrono::steady_clock::time_point{};
        active_surface_dirty_ = true;
    }

    bool collect_ready_jobs() {
        bool changed = false;
        uint32_t collected = 0u;
        for (size_t i = 0; i < jobs_.size();) {
            if (jobs_[i].future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                ++i;
                continue;
            }
            CaveInteriorBuildResult result = jobs_[i].future.get();
            queued_ids_.erase(jobs_[i].feature_id);
            jobs_.erase(jobs_.begin() + static_cast<std::ptrdiff_t>(i));
            if (result.generation == edit_generation_ &&
                !result.surface.vertices.empty() &&
                !result.surface.triangle_indices.empty()) {
                cache_[result.feature_id] = {
                    std::move(result.surface),
                    result.stats,
                    ++build_serial_,
                    result.build_ms,
                };
                stats_.last_build_ms = result.build_ms;
                active_surface_dirty_ = true;
                changed = true;
            }
            if (++collected >= 1u) {
                break;
            }
        }
        return changed;
    }

    bool choose_active_features(const ae::CameraView& view, ae::Vec3 aim_direction, uint32_t targeted_feature_id) {
        std::vector<uint32_t> previous = active_ids_;
        active_ids_.clear();
        if (features_.empty()) {
            return previous != active_ids_;
        }

        if (targeted_feature_id != 0u) {
            active_ids_.push_back(targeted_feature_id);
        }

        const ae::Vec3 surface_focus = ae::length(view.eye) > 0.000001f
            ? ae::normalize(view.eye)
            : ae::Vec3{0.0f, 1.0f, 0.0f};
        aim_direction = ae::normalize(aim_direction);

        std::vector<std::pair<float, uint32_t>> nearby;
        nearby.reserve(RenderedInteriorBudget);
        for (const ae::LocalVoxelFeature& feature : features_) {
            if (active_ids_.size() >= ActiveBudget) {
                break;
            }
            if (std::find(active_ids_.begin(), active_ids_.end(), feature.feature_id) != active_ids_.end()) {
                continue;
            }
            const float distance_km = cave_feature_stream_distance_km(feature, current_eye_mesh_);
            if (distance_km <= AlwaysRenderedRadiusKm) {
                nearby.push_back({distance_km, feature.feature_id});
            }
        }
        std::sort(nearby.begin(), nearby.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
        for (const auto& [distance_km, feature_id] : nearby) {
            (void)distance_km;
            if (active_ids_.size() >= ActiveBudget) {
                break;
            }
            active_ids_.push_back(feature_id);
        }

        for (const ae::LocalVoxelFeature& feature : features_) {
            if (active_ids_.size() >= ActiveBudget) {
                break;
            }
            if (!is_terrain_hole_feature_id(feature.feature_id) ||
                std::find(active_ids_.begin(), active_ids_.end(), feature.feature_id) != active_ids_.end()) {
                continue;
            }
            active_ids_.push_back(feature.feature_id);
        }

        std::vector<std::pair<float, uint32_t>> best;
        best.reserve(ActiveBudget);
        for (const ae::LocalVoxelFeature& feature : features_) {
            if (std::find(active_ids_.begin(), active_ids_.end(), feature.feature_id) != active_ids_.end()) {
                continue;
            }
            const float stream_distance = cave_feature_stream_distance_km(feature, current_eye_mesh_);
            const float proximity = ae::dot(surface_focus, feature.center_mesh);
            const float aim = std::max(0.0f, ae::dot(aim_direction, feature.center_mesh));
            const float distance_score = -std::max(stream_distance, 0.0f) * 0.012f;
            const float inside_bonus = stream_distance <= 0.0f ? 12.0f : 0.0f;
            const float score = inside_bonus + distance_score + proximity * 1.35f + aim * 0.28f;
            if (best.size() < ActiveBudget) {
                best.push_back({score, feature.feature_id});
                continue;
            }
            auto worst = std::min_element(best.begin(), best.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
            if (worst != best.end() && score > worst->first) {
                *worst = {score, feature.feature_id};
            }
        }

        const uint32_t remaining = ActiveBudget > active_ids_.size()
            ? ActiveBudget - static_cast<uint32_t>(active_ids_.size())
            : 0u;
        if (best.size() > remaining) {
            std::sort(best.begin(), best.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first > rhs.first;
            });
            best.resize(remaining);
        }
        std::sort(best.begin(), best.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        });
        for (const auto& [score, feature_id] : best) {
            (void)score;
            active_ids_.push_back(feature_id);
        }

        for (uint32_t feature_id : active_ids_) {
            auto found = cache_.find(feature_id);
            if (found != cache_.end()) {
                found->second.last_used = ++build_serial_;
            }
        }
        if (previous != active_ids_) {
            active_surface_dirty_ = true;
            return true;
        }
        return false;
    }

    void queue_missing_active_features() {
        for (uint32_t feature_id : active_ids_) {
            if (jobs_.size() >= worker_count_) {
                return;
            }
            if (cache_.find(feature_id) != cache_.end() || queued_ids_.find(feature_id) != queued_ids_.end()) {
                continue;
            }
            queue_feature(feature_id);
        }
    }

    void queue_feature(uint32_t feature_id) {
        if (feature_id == 0u || jobs_.size() >= worker_count_) {
            return;
        }
        auto feature_it = std::find_if(features_.begin(), features_.end(), [feature_id](const ae::LocalVoxelFeature& feature) {
            return feature.feature_id == feature_id;
        });
        if (feature_it == features_.end()) {
            return;
        }
        queued_ids_.insert(feature_id);
        ae::LocalVoxelFeature feature = *feature_it;
        ae::MarchingCubesConfig config = config_;
        const float grid_radius = grid_radius_;
        const uint32_t generation = edit_generation_;
        jobs_.push_back({
            feature_id,
            generation,
            std::async(std::launch::async, [feature, config, grid_radius, generation]() mutable {
                const auto begin = std::chrono::steady_clock::now();
                CaveInteriorBuildResult result;
                result.feature_id = feature.feature_id;
                result.generation = generation;
                result.surface = ae::build_cave_surface_net_for_feature(feature, config, grid_radius, &result.stats);
                result.build_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin
                ).count();
                return result;
            })
        });
    }

    void evict_cache() {
        if (cache_.size() <= CacheBudget) {
            return;
        }
        std::unordered_set<uint32_t> active_set(active_ids_.begin(), active_ids_.end());
        while (cache_.size() > CacheBudget) {
            auto victim = cache_.end();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (active_set.find(it->first) != active_set.end()) {
                    continue;
                }
                if (victim == cache_.end() || it->second.last_used < victim->second.last_used) {
                    victim = it;
                }
            }
            if (victim == cache_.end()) {
                break;
            }
            cache_.erase(victim);
        }
    }

    void rebuild_active_surface() {
        active_surface_ = {};
        active_holes_.clear();
        active_surface_.source_depth = config_.voxel_features.cave_depth;
        active_surface_.material_id = config_.surface_net_material_id;
        active_surface_.bounds_radius = grid_radius_;
        active_surface_.local_patch_depth = config_.voxel_features.cave_depth;
        struct RenderCandidate {
            float distance_km = 0.0f;
            uint32_t active_order = 0u;
            const ae::LocalVoxelFeature* feature = nullptr;
            const CachedInterior* cached = nullptr;
        };
        std::vector<RenderCandidate> render_candidates;
        render_candidates.reserve(active_ids_.size());
        for (uint32_t feature_id : active_ids_) {
            auto found = cache_.find(feature_id);
            if (found == cache_.end()) {
                continue;
            }
            const ae::LocalVoxelFeature* feature = nullptr;
            const size_t feature_index = static_cast<size_t>(feature_id - 1u);
            if (feature_index < features_.size() && features_[feature_index].feature_id == feature_id) {
                feature = &features_[feature_index];
            } else {
                auto feature_it = std::find_if(features_.begin(), features_.end(), [feature_id](const ae::LocalVoxelFeature& feature) {
                    return feature.feature_id == feature_id;
                });
                if (feature_it != features_.end()) {
                    feature = &*feature_it;
                }
            }
            if (feature == nullptr) {
                continue;
            }
            render_candidates.push_back({
                cave_feature_stream_distance_km(*feature, current_eye_mesh_),
                static_cast<uint32_t>(render_candidates.size()),
                feature,
                &found->second,
            });
        }
        std::sort(render_candidates.begin(), render_candidates.end(), [](const RenderCandidate& lhs, const RenderCandidate& rhs) {
            if (std::fabs(lhs.distance_km - rhs.distance_km) > 0.001f) {
                return lhs.distance_km < rhs.distance_km;
            }
            return lhs.active_order < rhs.active_order;
        });

        uint32_t rendered_count = 0u;
        for (const RenderCandidate& candidate : render_candidates) {
            if (rendered_count >= RenderedInteriorBudget) {
                break;
            }
            active_holes_.push_back(*candidate.feature);
            append_surface_mesh(active_surface_, candidate.cached->surface);
            ++rendered_count;
        }
        active_surface_.local_patch_count = static_cast<uint32_t>(active_ids_.size());
        active_surface_.local_vertex_count = static_cast<uint32_t>(active_surface_.vertices.size());
        active_surface_.local_triangle_count = static_cast<uint32_t>(active_surface_.triangle_indices.size() / 3u);
        stats_.rendered = rendered_count;
    }

    std::vector<ae::LocalVoxelFeature> features_;
    std::vector<ae::LocalVoxelFeature> base_features_;
    std::unordered_set<uint32_t> terrain_feature_ids_;
    size_t terrain_feature_count_ = 0u;
    ae::MarchingCubesConfig config_;
    float grid_radius_ = 1.0f;
    uint32_t worker_count_ = 1u;
    uint64_t build_serial_ = 0u;
    uint32_t edit_generation_ = 0u;
    uint32_t last_targeted_feature_id_ = 0u;
    ae::Vec3 current_eye_mesh_ = {};
    std::chrono::steady_clock::time_point last_rescore_ = {};
    std::vector<uint32_t> active_ids_;
    std::unordered_map<uint32_t, CachedInterior> cache_;
    std::unordered_set<uint32_t> queued_ids_;
    std::vector<CaveBuildJob> jobs_;
    ae::SurfaceNetMesh active_surface_;
    std::vector<ae::LocalVoxelFeature> active_holes_;
    CaveStreamStats stats_;
    bool active_surface_dirty_ = true;
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
    mesh_config.voxel_features.cave_count = 10000;
    mesh_config.voxel_features.cave_anchor_count = 10000;
    mesh_config.voxel_features.cave_depth = StreamedCaveDepth;
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
    CaveInteriorStreamer cave_streamer;
    cave_streamer.configure(current_mesh, mesh_config);
    renderer.update_cave_interiors(cave_streamer.active_surface());
    auto sync_terrain_hole_renderer = [&]() {
        renderer.update_terrain_holes(build_goldberg_clip_holes(cave_streamer.active_holes(), voxel_edits));
        renderer.update_cave_dig_transitions(cave_streamer.active_holes(), voxel_edits);
        renderer.update_terrain_height_masks(voxel_edits.terrain_masks);
    };
    sync_terrain_hole_renderer();
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
    uint32_t targeted_feature_id = 0u;
    std::chrono::steady_clock::time_point last_cave_target_update = {};
    bool terrain_paint_active = false;
    size_t active_terrain_mask_index = 0u;
    ae::Vec3 last_terrain_paint_hit = {};
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
                    mesh_config.voxel_features.cave_depth = StreamedCaveDepth;
                    mesh_config.svo_debug_draw_depth = SvoDebugDrawDepth;
                    request_voxel_mesh_rebuild("cave depth toggle");
                    std::cout << "Streamed cave interior depth " << mesh_config.voxel_features.cave_depth
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
                    voxel_edits.terrain_masks.clear();
                    terrain_paint_active = false;
                    mesh_config.voxel_edits = voxel_edits;
                    cave_streamer.configure(current_mesh, mesh_config);
                    renderer.update_cave_interiors(cave_streamer.active_surface());
                    sync_terrain_hole_renderer();
                    request_voxel_mesh_rebuild("clear dig edits");
                    std::cout << "Cleared dig edits; cave stream reset" << std::endl;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT) {
                const ae::CameraView dig_view = render_options.follow_ship ? make_ship_follow_view(spaceship) : make_free_camera_view(camera);
                const ae::Vec3 dig_direction = ae::normalize(dig_view.target - dig_view.eye);
                ae::Vec3 hit_mesh = {};
                uint32_t hit_feature_id = 0u;
                ae::Vec3 cave_surface_hit_mesh = {};
                float cave_surface_hit_t = std::numeric_limits<float>::max();
                const bool hit_active_cave = ray_surface_net_intersection(
                    cave_streamer.active_surface(),
                    dig_view.eye / ae::PlanetRadiusKilometers,
                    dig_direction,
                    cave_surface_hit_mesh,
                    &cave_surface_hit_t
                );
                if (hit_active_cave) {
                    hit_mesh = cave_surface_hit_mesh;
                    hit_feature_id = nearest_cave_feature_id_for_point(cave_streamer.active_holes(), hit_mesh);
                }
                bool hit_cave_feature = hit_active_cave;
                CaveFeatureRayHit cave_hit = {};
                if (!hit_cave_feature && ray_cave_feature_hit(
                        cave_streamer.features(),
                        dig_view.eye / ae::PlanetRadiusKilometers,
                        dig_direction,
                        cave_hit,
                        CaveClickTargetMaxDistanceKm
                    )) {
                    hit_mesh = cave_hit.hit_mesh;
                    hit_feature_id = cave_hit.feature_id;
                    hit_cave_feature = true;
                }
                if (hit_cave_feature) {
                    terrain_paint_active = false;
                    voxel_edits.digs.push_back({
                        hit_mesh,
                        8.0f,
                        voxel_edits.local_depth,
                        ae::VoxelDigTarget::CaveInterior,
                        hit_feature_id,
                    });
                    mesh_config.voxel_edits = voxel_edits;
                    cave_streamer.update_config(mesh_config);
                    mesh_config.lod_camera_position = dig_view.eye;
                    if (hit_feature_id != 0u) {
                        cave_streamer.invalidate_feature(hit_feature_id);
                    } else if (hit_active_cave) {
                        cave_streamer.invalidate_active();
                    }
                    sync_terrain_hole_renderer();
                    std::cout << "Dig edit " << voxel_edits.digs.size()
                              << ": radius 8 km, local depth " << voxel_edits.local_depth
                              << " cave stream rebuild requested" << std::endl;
                } else if (ray_quantized_mesh_intersection(
                               current_mesh,
                               dig_view.eye / ae::PlanetRadiusKilometers,
                               dig_direction,
                               hit_mesh)) {
                    active_terrain_mask_index = terrain_mask_for_point(voxel_edits, hit_mesh);
                    paint_terrain_mask_point(voxel_edits, active_terrain_mask_index, hit_mesh, 8.0f);
                    last_terrain_paint_hit = hit_mesh;
                    terrain_paint_active = true;
                    mesh_config.voxel_edits = voxel_edits;
                    cave_streamer.update_config(mesh_config);
                    mesh_config.lod_camera_position = dig_view.eye;
                    sync_terrain_hole_renderer();
                    std::cout << "Terrain height mask stroke: brush 8 km, masks "
                              << voxel_edits.terrain_masks.size()
                              << ", streamed cave requested" << std::endl;
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT) {
                terrain_paint_active = false;
            } else if (event.type == SDL_EVENT_MOUSE_MOTION && terrain_paint_active && (event.motion.state & SDL_BUTTON_LMASK) != 0) {
                const ae::CameraView paint_view = render_options.follow_ship ? make_ship_follow_view(spaceship) : make_free_camera_view(camera);
                const ae::Vec3 paint_direction = ae::normalize(paint_view.target - paint_view.eye);
                ae::Vec3 paint_hit_mesh = {};
                if (ray_quantized_mesh_intersection(
                        current_mesh,
                        paint_view.eye / ae::PlanetRadiusKilometers,
                        paint_direction,
                        paint_hit_mesh)) {
                    float projected_x = 0.0f;
                    float projected_y = 0.0f;
                    if (active_terrain_mask_index >= voxel_edits.terrain_masks.size() ||
                        !terrain_mask_project(voxel_edits.terrain_masks[active_terrain_mask_index], paint_hit_mesh, projected_x, projected_y)) {
                        active_terrain_mask_index = terrain_mask_for_point(voxel_edits, paint_hit_mesh);
                        last_terrain_paint_hit = paint_hit_mesh;
                    }
                    paint_terrain_mask_line(
                        voxel_edits.terrain_masks[active_terrain_mask_index],
                        last_terrain_paint_hit,
                        paint_hit_mesh,
                        8.0f,
                        TerrainHeightHole
                    );
                    last_terrain_paint_hit = paint_hit_mesh;
                    mesh_config.voxel_edits = voxel_edits;
                    cave_streamer.update_config(mesh_config);
                    mesh_config.lod_camera_position = paint_view.eye;
                    sync_terrain_hole_renderer();
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
        const ae::Vec3 frame_aim_direction = ae::normalize(camera_view.target - camera_view.eye);
        const auto cave_target_now = std::chrono::steady_clock::now();
        if (last_cave_target_update == std::chrono::steady_clock::time_point{} ||
            cave_target_now - last_cave_target_update >= std::chrono::milliseconds(125)) {
            CaveFeatureRayHit targeted_cave = {};
            targeted_feature_id = ray_cave_feature_hit(
                cave_streamer.features(),
                camera_view.eye / ae::PlanetRadiusKilometers,
                frame_aim_direction,
                targeted_cave,
                CaveViewTargetMaxDistanceKm
            ) ? targeted_cave.feature_id : 0u;
            last_cave_target_update = cave_target_now;
        }
        if (cave_streamer.update(camera_view, frame_aim_direction, targeted_feature_id)) {
            renderer.update_cave_interiors(cave_streamer.active_surface());
            sync_terrain_hole_renderer();
        }
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
                const bool reset_cave_streamer = result.mode == MeshBuildMode::Full || result.reason != "terrain dig";
                if (reset_cave_streamer) {
                    cave_streamer.configure(current_mesh, mesh_config);
                }
                renderer.update_cave_interiors(cave_streamer.active_surface());
                sync_terrain_hole_renderer();
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
                          << " MiB, cave anchors " << result.mesh.cave_anchor_points.size()
                          << ", cave descriptors " << result.mesh.voxel_features.size()
                          << ", streamed active " << cave_streamer.stats().active
                          << ", cached " << cave_streamer.stats().cached
                          << ", queued " << cave_streamer.stats().queued << std::endl;
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
                const char* voxel_mode = "off";
                if (render_options.show_voxels) {
                    if (renderer_stats.voxel_compute_used) {
                        voxel_mode = "compute";
                    } else if (!renderer_stats.voxel_compute_available) {
                        voxel_mode = "cpu";
                    } else if (renderer_stats.voxel_compute_fallback_code == 2u) {
                        voxel_mode = "waiting-svo";
                    } else if (renderer_stats.voxel_compute_fallback_code == 3u) {
                        voxel_mode = "bad-buffer";
                    } else if (renderer_stats.voxel_compute_fallback_code == 4u) {
                        voxel_mode = "gl-error";
                    } else {
                        voxel_mode = "cpu-fallback";
                    }
                }
                const CaveStreamStats& cave_stats = cave_streamer.stats();
                char title_buffer[320] = {};
                std::snprintf(
                    title_buffer,
                    sizeof(title_buffer),
                    "AETHERONUS - FPS %.0f | frame %.2f ms worst %.2f | render %.2f ms | GPU mesh %.2f surf %.2f dbg %.2f | caves %u/%u r%u cache %u q %u %.1f ms | vox %s %.2f ms | draws %u",
                    displayed_fps,
                    1000.0f / std::max(displayed_fps, 0.001f),
                    recent_worst_frame_ms,
                    renderer_stats.render_cpu_ms,
                    renderer_stats.gpu_mesh_ms,
                    renderer_stats.gpu_surface_net_ms,
                    renderer_stats.gpu_debug_ms,
                    cave_stats.active,
                    cave_stats.descriptors,
                    cave_stats.rendered,
                    cave_stats.cached,
                    cave_stats.queued,
                    cave_stats.last_build_ms,
                    voxel_mode,
                    renderer_stats.voxel_compute_dispatch_ms,
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
