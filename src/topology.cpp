#include "aetheronus/topology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace ae {
namespace {

uint64_t edge_key(uint32_t a, uint32_t b) {
    const uint32_t low = std::min(a, b);
    const uint32_t high = std::max(a, b);
    return (static_cast<uint64_t>(low) << 32) | high;
}

Vec3 triangle_center(const std::vector<Vec3>& vertices, const Triangle& triangle) {
    return normalize((vertices[triangle.a] + vertices[triangle.b] + vertices[triangle.c]) / 3.0f);
}

struct DirectionKey {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;

    bool operator==(const DirectionKey& rhs) const {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct DirectionKeyHash {
    size_t operator()(DirectionKey key) const {
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&](int64_t value) {
            uint64_t word = static_cast<uint64_t>(value);
            hash ^= word;
            hash *= 1099511628211ull;
        };
        mix(key.x);
        mix(key.y);
        mix(key.z);
        return static_cast<size_t>(hash);
    }
};

struct LatticePoint {
    int32_t i = 0;
    int32_t j = 0;

    bool operator==(const LatticePoint& rhs) const {
        return i == rhs.i && j == rhs.j;
    }
};

struct LatticePointHash {
    size_t operator()(LatticePoint point) const {
        const uint64_t packed =
            (static_cast<uint64_t>(static_cast<uint32_t>(point.i)) << 32) |
            static_cast<uint32_t>(point.j);
        return static_cast<size_t>(packed ^ (packed >> 33u));
    }
};

struct LocalGoldbergPoint {
    LatticePoint lattice;
    Vec2 plane;
    uint32_t geodesic_index = 0;
};

struct LocalTriangulationPoint {
    Vec2 plane;
};

struct LocalTriangle {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
};

struct TriangleKey {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;

    bool operator==(const TriangleKey& rhs) const {
        return a == rhs.a && b == rhs.b && c == rhs.c;
    }
};

struct TriangleKeyHash {
    size_t operator()(TriangleKey key) const {
        uint64_t hash = 1469598103934665603ull;
        auto mix = [&](uint32_t value) {
            hash ^= value;
            hash *= 1099511628211ull;
        };
        mix(key.a);
        mix(key.b);
        mix(key.c);
        return static_cast<size_t>(hash);
    }
};

GeodesicMesh build_icosahedron() {
    const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;

    GeodesicMesh mesh;
    mesh.vertices = {
        normalize({-1.0f, phi, 0.0f}),
        normalize({1.0f, phi, 0.0f}),
        normalize({-1.0f, -phi, 0.0f}),
        normalize({1.0f, -phi, 0.0f}),
        normalize({0.0f, -1.0f, phi}),
        normalize({0.0f, 1.0f, phi}),
        normalize({0.0f, -1.0f, -phi}),
        normalize({0.0f, 1.0f, -phi}),
        normalize({phi, 0.0f, -1.0f}),
        normalize({phi, 0.0f, 1.0f}),
        normalize({-phi, 0.0f, -1.0f}),
        normalize({-phi, 0.0f, 1.0f}),
    };
    mesh.original_vertex_count = static_cast<uint32_t>(mesh.vertices.size());
    mesh.triangles = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1},
    };
    return mesh;
}

DirectionKey direction_key(Vec3 direction) {
    constexpr double Scale = 100000.0;
    return {
        static_cast<int64_t>(std::llround(static_cast<double>(direction.x) * Scale)),
        static_cast<int64_t>(std::llround(static_cast<double>(direction.y) * Scale)),
        static_cast<int64_t>(std::llround(static_cast<double>(direction.z) * Scale)),
    };
}

bool lattice_inside_or_on(LatticePoint point, GoldbergConfig config, float tolerance = 0.00001f) {
    const float f = static_cast<float>(config.frequency());
    if (f <= 0.0f) {
        return false;
    }

    const float i = static_cast<float>(point.i);
    const float j = static_cast<float>(point.j);
    const float alpha = (static_cast<float>(config.m + config.n) * i + static_cast<float>(config.n) * j) / f;
    const float beta = (-static_cast<float>(config.n) * i + static_cast<float>(config.m) * j) / f;
    const float gamma = 1.0f - alpha - beta;
    return alpha >= -tolerance && beta >= -tolerance && gamma >= -tolerance;
}

std::array<float, 3> lattice_barycentric(float i, float j, GoldbergConfig config) {
    const float f = static_cast<float>(config.frequency());
    const float beta = (static_cast<float>(config.m + config.n) * i + static_cast<float>(config.n) * j) / f;
    const float gamma = (-static_cast<float>(config.n) * i + static_cast<float>(config.m) * j) / f;
    const float alpha = 1.0f - beta - gamma;
    return {alpha, beta, gamma};
}

bool barycentric_inside_or_on(std::array<float, 3> barycentric, float tolerance = 0.00001f) {
    return barycentric[0] >= -tolerance &&
           barycentric[1] >= -tolerance &&
           barycentric[2] >= -tolerance;
}

Vec2 lattice_to_plane(LatticePoint point) {
    constexpr float Sqrt3Over2 = 0.86602540378443864676f;
    return {
        static_cast<float>(point.i) + 0.5f * static_cast<float>(point.j),
        Sqrt3Over2 * static_cast<float>(point.j),
    };
}

Vec3 lattice_to_sphere_point(
    LatticePoint point,
    GoldbergConfig config,
    Vec3 a,
    Vec3 b,
    Vec3 c
) {
    const float f = static_cast<float>(config.frequency());
    const float i = static_cast<float>(point.i);
    const float j = static_cast<float>(point.j);
    const float beta = (static_cast<float>(config.m + config.n) * i + static_cast<float>(config.n) * j) / f;
    const float gamma = (-static_cast<float>(config.n) * i + static_cast<float>(config.m) * j) / f;
    const float alpha = 1.0f - beta - gamma;
    return normalize(a * alpha + b * beta + c * gamma);
}

uint32_t adjacent_face_index(const GeodesicMesh& base, uint32_t face_index, uint32_t edge_a, uint32_t edge_b) {
    for (uint32_t candidate = 0; candidate < base.triangles.size(); ++candidate) {
        if (candidate == face_index) {
            continue;
        }
        const Triangle& triangle = base.triangles[candidate];
        const std::array<uint32_t, 3> vertices = {triangle.a, triangle.b, triangle.c};
        const bool has_a = std::find(vertices.begin(), vertices.end(), edge_a) != vertices.end();
        const bool has_b = std::find(vertices.begin(), vertices.end(), edge_b) != vertices.end();
        if (has_a && has_b) {
            return candidate;
        }
    }
    return face_index;
}

uint32_t third_face_vertex(const Triangle& triangle, uint32_t edge_a, uint32_t edge_b) {
    if (triangle.a != edge_a && triangle.a != edge_b) return triangle.a;
    if (triangle.b != edge_a && triangle.b != edge_b) return triangle.b;
    return triangle.c;
}

Vec3 folded_lattice_to_sphere_point(
    const GeodesicMesh& base,
    uint32_t face_index,
    LatticePoint point,
    GoldbergConfig config
) {
    std::array<float, 3> weights = lattice_barycentric(
        static_cast<float>(point.i),
        static_cast<float>(point.j),
        config
    );

    for (uint32_t fold = 0; fold < 16u && !barycentric_inside_or_on(weights); ++fold) {
        const Triangle& face = base.triangles[face_index];
        std::array<uint32_t, 2> edge = {};
        std::unordered_map<uint32_t, float> next_weights;

        if (weights[0] < -0.00001f) {
            edge = {face.b, face.c};
            const uint32_t next_face = adjacent_face_index(base, face_index, edge[0], edge[1]);
            const uint32_t opposite = third_face_vertex(base.triangles[next_face], edge[0], edge[1]);
            next_weights[edge[0]] = weights[1] + weights[0];
            next_weights[edge[1]] = weights[2] + weights[0];
            next_weights[opposite] = -weights[0];
            face_index = next_face;
        } else if (weights[1] < -0.00001f) {
            edge = {face.c, face.a};
            const uint32_t next_face = adjacent_face_index(base, face_index, edge[0], edge[1]);
            const uint32_t opposite = third_face_vertex(base.triangles[next_face], edge[0], edge[1]);
            next_weights[edge[0]] = weights[2] + weights[1];
            next_weights[edge[1]] = weights[0] + weights[1];
            next_weights[opposite] = -weights[1];
            face_index = next_face;
        } else if (weights[2] < -0.00001f) {
            edge = {face.a, face.b};
            const uint32_t next_face = adjacent_face_index(base, face_index, edge[0], edge[1]);
            const uint32_t opposite = third_face_vertex(base.triangles[next_face], edge[0], edge[1]);
            next_weights[edge[0]] = weights[0] + weights[2];
            next_weights[edge[1]] = weights[1] + weights[2];
            next_weights[opposite] = -weights[2];
            face_index = next_face;
        }

        const Triangle& next_face = base.triangles[face_index];
        weights = {
            next_weights[next_face.a],
            next_weights[next_face.b],
            next_weights[next_face.c],
        };
    }

    const float sum = std::max(0.000001f, weights[0] + weights[1] + weights[2]);
    weights = {
        std::max(0.0f, weights[0]) / sum,
        std::max(0.0f, weights[1]) / sum,
        std::max(0.0f, weights[2]) / sum,
    };

    const Triangle& face = base.triangles[face_index];
    return normalize(
        base.vertices[face.a] * weights[0] +
        base.vertices[face.b] * weights[1] +
        base.vertices[face.c] * weights[2]
    );
}

float signed_area2(Vec2 a, Vec2 b, Vec2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool point_in_circumcircle(Vec2 point, Vec2 a, Vec2 b, Vec2 c) {
    const double ax = static_cast<double>(a.x) - static_cast<double>(point.x);
    const double ay = static_cast<double>(a.y) - static_cast<double>(point.y);
    const double bx = static_cast<double>(b.x) - static_cast<double>(point.x);
    const double by = static_cast<double>(b.y) - static_cast<double>(point.y);
    const double cx = static_cast<double>(c.x) - static_cast<double>(point.x);
    const double cy = static_cast<double>(c.y) - static_cast<double>(point.y);

    const double determinant =
        (ax * ax + ay * ay) * (bx * cy - by * cx) -
        (bx * bx + by * by) * (ax * cy - ay * cx) +
        (cx * cx + cy * cy) * (ax * by - ay * bx);
    const float orientation = signed_area2(a, b, c);
    return orientation > 0.0f ? determinant > 0.000000001 : determinant < -0.000000001;
}

std::vector<LocalTriangle> triangulate_local_points(const std::vector<LocalGoldbergPoint>& local_points) {
    if (local_points.size() < 3u) {
        return {};
    }

    std::vector<LocalTriangulationPoint> points;
    points.reserve(local_points.size() + 3u);
    Vec2 min_bounds = local_points.front().plane;
    Vec2 max_bounds = local_points.front().plane;
    for (const LocalGoldbergPoint& point : local_points) {
        min_bounds.x = std::min(min_bounds.x, point.plane.x);
        min_bounds.y = std::min(min_bounds.y, point.plane.y);
        max_bounds.x = std::max(max_bounds.x, point.plane.x);
        max_bounds.y = std::max(max_bounds.y, point.plane.y);
        points.push_back({point.plane});
    }

    const float span_x = std::max(1.0f, max_bounds.x - min_bounds.x);
    const float span_y = std::max(1.0f, max_bounds.y - min_bounds.y);
    const float span = std::max(span_x, span_y) * 16.0f;
    const Vec2 center = {
        (min_bounds.x + max_bounds.x) * 0.5f,
        (min_bounds.y + max_bounds.y) * 0.5f,
    };

    const uint32_t super_a = static_cast<uint32_t>(points.size());
    points.push_back({{center.x - span, center.y - span}});
    const uint32_t super_b = static_cast<uint32_t>(points.size());
    points.push_back({{center.x + span, center.y - span}});
    const uint32_t super_c = static_cast<uint32_t>(points.size());
    points.push_back({{center.x, center.y + span}});

    std::vector<LocalTriangle> triangles = {{super_a, super_b, super_c}};
    auto make_triangle = [&](uint32_t a, uint32_t b, uint32_t c) {
        if (signed_area2(points[a].plane, points[b].plane, points[c].plane) < 0.0f) {
            std::swap(b, c);
        }
        return LocalTriangle{a, b, c};
    };

    for (uint32_t point_index = 0; point_index < local_points.size(); ++point_index) {
        std::vector<LocalTriangle> kept;
        std::vector<std::array<uint32_t, 2>> boundary_edges;

        for (const LocalTriangle& triangle : triangles) {
            if (!point_in_circumcircle(
                    points[point_index].plane,
                    points[triangle.a].plane,
                    points[triangle.b].plane,
                    points[triangle.c].plane)) {
                kept.push_back(triangle);
                continue;
            }

            const std::array<std::array<uint32_t, 2>, 3> edges = {{
                {{triangle.a, triangle.b}},
                {{triangle.b, triangle.c}},
                {{triangle.c, triangle.a}},
            }};
            for (std::array<uint32_t, 2> edge : edges) {
                const auto reversed = std::find_if(boundary_edges.begin(), boundary_edges.end(), [&](std::array<uint32_t, 2> other) {
                    return other[0] == edge[1] && other[1] == edge[0];
                });
                if (reversed != boundary_edges.end()) {
                    boundary_edges.erase(reversed);
                } else {
                    boundary_edges.push_back(edge);
                }
            }
        }

        for (std::array<uint32_t, 2> edge : boundary_edges) {
            kept.push_back(make_triangle(edge[0], edge[1], point_index));
        }
        triangles = std::move(kept);
    }

    std::vector<LocalTriangle> result;
    result.reserve(triangles.size());
    const uint32_t real_count = static_cast<uint32_t>(local_points.size());
    for (LocalTriangle triangle : triangles) {
        if (triangle.a >= real_count || triangle.b >= real_count || triangle.c >= real_count) {
            continue;
        }
        result.push_back(make_triangle(triangle.a, triangle.b, triangle.c));
    }
    return result;
}

uint32_t geodesic_vertex_index(
    GeodesicMesh& mesh,
    std::unordered_map<DirectionKey, uint32_t, DirectionKeyHash>& vertex_lookup,
    Vec3 direction
) {
    const DirectionKey key = direction_key(direction);
    const auto found = vertex_lookup.find(key);
    if (found != vertex_lookup.end()) {
        return found->second;
    }

    const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(direction);
    vertex_lookup.emplace(key, index);
    return index;
}

void add_oriented_triangle(GeodesicMesh& mesh, uint32_t a, uint32_t b, uint32_t c) {
    if (a == b || b == c || c == a) {
        return;
    }

    const Vec3 pa = mesh.vertices[a];
    const Vec3 pb = mesh.vertices[b];
    const Vec3 pc = mesh.vertices[c];
    const Vec3 outward = normalize((pa + pb + pc) / 3.0f);
    if (dot(cross(pb - pa, pc - pa), outward) < 0.0f) {
        std::swap(b, c);
    }
    mesh.triangles.push_back({a, b, c});
}

TriangleKey triangle_key(Triangle triangle) {
    std::array<uint32_t, 3> values = {triangle.a, triangle.b, triangle.c};
    std::sort(values.begin(), values.end());
    return {values[0], values[1], values[2]};
}

void deduplicate_triangles(GeodesicMesh& mesh) {
    std::unordered_set<TriangleKey, TriangleKeyHash> seen;
    std::vector<Triangle> unique;
    unique.reserve(mesh.triangles.size());
    for (const Triangle& triangle : mesh.triangles) {
        const TriangleKey key = triangle_key(triangle);
        if (seen.insert(key).second) {
            unique.push_back(triangle);
        }
    }
    mesh.triangles = std::move(unique);
}

GeodesicMesh build_goldberg_geodesic(GoldbergConfig config) {
    if (config.frequency() == 0u) {
        config = {};
    }

    const GeodesicMesh base = build_icosahedron();
    GeodesicMesh mesh;
    mesh.original_vertex_count = base.original_vertex_count;
    mesh.vertices.reserve(10u * config.frequency() + 2u);
    mesh.triangles.reserve(20u * config.frequency());

    std::unordered_map<DirectionKey, uint32_t, DirectionKeyHash> vertex_lookup;
    const int32_t min_i = std::min({0, static_cast<int32_t>(config.m), -static_cast<int32_t>(config.n)}) - 2;
    const int32_t max_i = std::max({0, static_cast<int32_t>(config.m), -static_cast<int32_t>(config.n)}) + 2;
    const int32_t min_j = std::min({0, static_cast<int32_t>(config.n), static_cast<int32_t>(config.m + config.n)}) - 2;
    const int32_t max_j = std::max({0, static_cast<int32_t>(config.n), static_cast<int32_t>(config.m + config.n)}) + 2;

    for (uint32_t face_index = 0; face_index < base.triangles.size(); ++face_index) {
        for (int32_t i = min_i; i <= max_i; ++i) {
            for (int32_t j = min_j; j <= max_j; ++j) {
                const std::array<std::array<LatticePoint, 3>, 2> candidates = {{
                    {{{i, j}, {i + 1, j}, {i, j + 1}}},
                    {{{i + 1, j}, {i + 1, j + 1}, {i, j + 1}}},
                }};

                for (const std::array<LatticePoint, 3>& local_triangle : candidates) {
                    const float center_i =
                        (static_cast<float>(local_triangle[0].i) +
                         static_cast<float>(local_triangle[1].i) +
                         static_cast<float>(local_triangle[2].i)) / 3.0f;
                    const float center_j =
                        (static_cast<float>(local_triangle[0].j) +
                         static_cast<float>(local_triangle[1].j) +
                         static_cast<float>(local_triangle[2].j)) / 3.0f;
                    if (!barycentric_inside_or_on(lattice_barycentric(center_i, center_j, config))) {
                        continue;
                    }

                    const uint32_t a = geodesic_vertex_index(
                        mesh,
                        vertex_lookup,
                        folded_lattice_to_sphere_point(base, face_index, local_triangle[0], config)
                    );
                    const uint32_t b = geodesic_vertex_index(
                        mesh,
                        vertex_lookup,
                        folded_lattice_to_sphere_point(base, face_index, local_triangle[1], config)
                    );
                    const uint32_t c = geodesic_vertex_index(
                        mesh,
                        vertex_lookup,
                        folded_lattice_to_sphere_point(base, face_index, local_triangle[2], config)
                    );
                    add_oriented_triangle(mesh, a, b, c);
                }
            }
        }
    }

    deduplicate_triangles(mesh);
    return mesh;
}

void add_unique(std::vector<uint32_t>& values, uint32_t value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

} // namespace

GoldbergTopology build_goldberg_topology(GoldbergConfig config) {
    GoldbergTopology topology;
    if (config.frequency() == 0u) {
        config = {};
    }
    topology.config = config;
    topology.frequency = config.frequency();
    topology.geodesic = build_goldberg_geodesic(config);

    topology.vertices.reserve(topology.geodesic.triangles.size());
    for (const Triangle& triangle : topology.geodesic.triangles) {
        topology.vertices.push_back({triangle_center(topology.geodesic.vertices, triangle)});
    }

    topology.cells.resize(topology.geodesic.vertices.size());
    for (uint32_t i = 0; i < topology.cells.size(); ++i) {
        GoldbergCell& cell = topology.cells[i];
        cell.center = topology.geodesic.vertices[i];
        cell.normal = topology.geodesic.vertices[i];
    }

    std::unordered_set<uint64_t> edges;
    for (uint32_t triangle_index = 0; triangle_index < topology.geodesic.triangles.size(); ++triangle_index) {
        const Triangle& triangle = topology.geodesic.triangles[triangle_index];
        const std::array<uint32_t, 3> vertices = {triangle.a, triangle.b, triangle.c};
        for (uint32_t vertex : vertices) {
            topology.cells[vertex].corner_indices.push_back(triangle_index);
        }

        for (int i = 0; i < 3; ++i) {
            const uint32_t a = vertices[i];
            const uint32_t b = vertices[(i + 1) % 3];
            const uint64_t key = edge_key(a, b);
            if (edges.insert(key).second) {
                add_unique(topology.cells[a].neighbor_indices, b);
                add_unique(topology.cells[b].neighbor_indices, a);
            }
        }
    }

    topology.edge_count = static_cast<uint32_t>(edges.size());

    for (GoldbergCell& cell : topology.cells) {
        cell.kind = cell.corner_indices.size() == 5 ? GoldbergCellKind::Pentagon : GoldbergCellKind::Hexagon;
        const Vec3 normal = cell.normal;
        const Vec3 reference_axis = std::fabs(normal.y) < 0.92f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
        const Vec3 tangent = normalize(cross(reference_axis, normal));
        const Vec3 bitangent = cross(normal, tangent);

        std::sort(cell.corner_indices.begin(), cell.corner_indices.end(), [&](uint32_t lhs, uint32_t rhs) {
            const Vec3 l = topology.vertices[lhs].position;
            const Vec3 r = topology.vertices[rhs].position;
            const float la = std::atan2(dot(l, bitangent), dot(l, tangent));
            const float ra = std::atan2(dot(r, bitangent), dot(r, tangent));
            return la < ra;
        });

        std::sort(cell.neighbor_indices.begin(), cell.neighbor_indices.end());
    }

    return topology;
}

TopologyValidation validate_topology(const GoldbergTopology& topology, uint32_t point_count) {
    std::ostringstream report;
    bool ok = true;

    auto require = [&](bool condition, const char* message) {
        if (!condition) {
            ok = false;
            report << "Validation failed: " << message << '\n';
        }
    };

    uint32_t pentagons = 0;
    uint32_t hexagons = 0;
    for (uint32_t i = 0; i < topology.cells.size(); ++i) {
        const GoldbergCell& cell = topology.cells[i];
        if (cell.kind == GoldbergCellKind::Pentagon) {
            ++pentagons;
        } else {
            ++hexagons;
        }

        require(cell.corner_indices.size() == 5 || cell.corner_indices.size() == 6, "cell corner count is not 5 or 6");
        require(cell.neighbor_indices.size() == cell.corner_indices.size(), "cell neighbor count does not match corner count");

        for (uint32_t neighbor : cell.neighbor_indices) {
            require(neighbor < topology.cells.size(), "neighbor index is out of range");
            if (neighbor < topology.cells.size()) {
                const auto& neighbors = topology.cells[neighbor].neighbor_indices;
                require(
                    std::find(neighbors.begin(), neighbors.end(), i) != neighbors.end(),
                    "neighbor link is not reciprocal"
                );
            }
        }
    }

    for (const Vec3& vertex : topology.geodesic.vertices) {
        require(std::fabs(length(vertex) - 1.0f) < 0.0005f, "geodesic vertex is not on the unit sphere");
    }
    for (const GoldbergVertex& vertex : topology.vertices) {
        require(std::fabs(length(vertex.position) - 1.0f) < 0.0005f, "Goldberg vertex is not on the unit sphere");
    }

    const uint32_t frequency = topology.frequency > 0u ? topology.frequency : topology.config.frequency();
    require(frequency > 0u, "Goldberg frequency is zero");
    const uint32_t expected_cells = 10u * frequency + 2u;
    const uint32_t expected_triangles = 20u * frequency;
    const uint32_t expected_edges = 30u * frequency;
    const uint32_t expected_hexagons = expected_cells >= 12u ? expected_cells - 12u : 0u;

    require(topology.geodesic.vertices.size() == expected_cells, "geodesic vertex count does not match Goldberg frequency");
    require(topology.geodesic.triangles.size() == expected_triangles, "geodesic triangle count does not match Goldberg frequency");
    require(topology.edge_count == expected_edges, "geodesic edge count does not match Goldberg frequency");
    require(topology.cells.size() == expected_cells, "Goldberg cell count does not match Goldberg frequency");
    require(pentagons == 12, "pentagon count is not 12");
    require(hexagons == expected_hexagons, "hexagon count does not match Goldberg frequency");
    require(point_count >= topology.cells.size(), "point count is less than cell count");

    if (ok) {
        report << "Topology validation OK: Goldberg("
               << topology.config.m << ',' << topology.config.n << "), frequency "
               << frequency << ", "
               << topology.geodesic.vertices.size() << " geodesic vertices, "
               << topology.geodesic.triangles.size() << " triangles, "
               << topology.edge_count << " edges, "
               << topology.cells.size() << " Goldberg cells ("
               << pentagons << " pentagons, " << hexagons << " hexagons), "
               << point_count << " point samples.";
    }

    return {ok, report.str()};
}

} // namespace ae
