#include "aetheronus/topology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

uint32_t midpoint_index(
    GeodesicMesh& mesh,
    std::unordered_map<uint64_t, uint32_t>& midpoint_cache,
    uint32_t a,
    uint32_t b
) {
    const uint64_t key = edge_key(a, b);
    const auto found = midpoint_cache.find(key);
    if (found != midpoint_cache.end()) {
        return found->second;
    }

    const Vec3 midpoint = normalize((mesh.vertices[a] + mesh.vertices[b]) * 0.5f);
    const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(midpoint);
    midpoint_cache.emplace(key, index);
    return index;
}

void subdivide(GeodesicMesh& mesh) {
    std::unordered_map<uint64_t, uint32_t> midpoint_cache;
    std::vector<Triangle> next;
    next.reserve(mesh.triangles.size() * 4);

    for (const Triangle& triangle : mesh.triangles) {
        const uint32_t ab = midpoint_index(mesh, midpoint_cache, triangle.a, triangle.b);
        const uint32_t bc = midpoint_index(mesh, midpoint_cache, triangle.b, triangle.c);
        const uint32_t ca = midpoint_index(mesh, midpoint_cache, triangle.c, triangle.a);

        next.push_back({triangle.a, ab, ca});
        next.push_back({triangle.b, bc, ab});
        next.push_back({triangle.c, ca, bc});
        next.push_back({ab, bc, ca});
    }

    mesh.triangles = std::move(next);
}

void add_unique(std::vector<uint32_t>& values, uint32_t value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

} // namespace

GoldbergTopology build_goldberg_topology(uint32_t subdivision_level) {
    GoldbergTopology topology;
    topology.geodesic = build_icosahedron();
    for (uint32_t i = 0; i < subdivision_level; ++i) {
        subdivide(topology.geodesic);
    }

    topology.vertices.reserve(topology.geodesic.triangles.size());
    for (const Triangle& triangle : topology.geodesic.triangles) {
        topology.vertices.push_back({triangle_center(topology.geodesic.vertices, triangle)});
    }

    topology.cells.resize(topology.geodesic.vertices.size());
    for (uint32_t i = 0; i < topology.cells.size(); ++i) {
        GoldbergCell& cell = topology.cells[i];
        cell.center = topology.geodesic.vertices[i];
        cell.normal = topology.geodesic.vertices[i];
        cell.kind = i < topology.geodesic.original_vertex_count ? GoldbergCellKind::Pentagon : GoldbergCellKind::Hexagon;
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

    require(topology.geodesic.vertices.size() == 162, "geodesic vertex count is not 162");
    require(topology.geodesic.triangles.size() == 320, "geodesic triangle count is not 320");
    require(topology.edge_count == 480, "geodesic edge count is not 480");
    require(topology.cells.size() == 162, "Goldberg cell count is not 162");
    require(pentagons == 12, "pentagon count is not 12");
    require(hexagons == 150, "hexagon count is not 150");
    require(point_count == topology.cells.size() * 4, "point count is not cell_count * 4");

    if (ok) {
        report << "Topology validation OK: "
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
