#pragma once

#include "aetheronus/math.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ae {

struct Triangle {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
};

struct GeodesicMesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    uint32_t original_vertex_count = 0;
};

enum class GoldbergCellKind : uint8_t {
    Pentagon,
    Hexagon,
};

struct GoldbergVertex {
    Vec3 position;
};

struct GoldbergCell {
    Vec3 center;
    Vec3 normal;
    std::vector<uint32_t> corner_indices;
    std::vector<uint32_t> neighbor_indices;
    GoldbergCellKind kind = GoldbergCellKind::Hexagon;
};

struct GoldbergTopology {
    GeodesicMesh geodesic;
    std::vector<GoldbergVertex> vertices;
    std::vector<GoldbergCell> cells;
    uint32_t edge_count = 0;
};

struct TopologyValidation {
    bool ok = false;
    std::string message;
};

GoldbergTopology build_goldberg_topology(uint32_t subdivision_level);
TopologyValidation validate_topology(const GoldbergTopology& topology, uint32_t point_count);

} // namespace ae
