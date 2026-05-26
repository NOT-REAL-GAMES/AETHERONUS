#include "aetheronus/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#include <immintrin.h>
#endif

namespace ae {
namespace {

struct PointOwnershipResolver {
    static constexpr uint32_t MaxCandidateCount = 8;

    std::vector<float> center_x;
    std::vector<float> center_y;
    std::vector<float> center_z;
    std::vector<std::array<uint32_t, MaxCandidateCount>> candidates_by_cell;
    std::vector<uint8_t> candidate_counts;
};

PointOwnershipResolver build_ownership_resolver(const GoldbergTopology& topology) {
    PointOwnershipResolver resolver;
    resolver.center_x.reserve(topology.cells.size());
    resolver.center_y.reserve(topology.cells.size());
    resolver.center_z.reserve(topology.cells.size());
    resolver.candidates_by_cell.resize(topology.cells.size());
    resolver.candidate_counts.resize(topology.cells.size(), 0);

    for (const GoldbergCell& cell : topology.cells) {
        resolver.center_x.push_back(cell.center.x);
        resolver.center_y.push_back(cell.center.y);
        resolver.center_z.push_back(cell.center.z);
    }

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        std::array<uint32_t, PointOwnershipResolver::MaxCandidateCount> candidates = {};
        uint32_t count = 0;
        auto append_candidate = [&](uint32_t candidate) {
            if (candidate >= topology.cells.size() || count >= candidates.size()) {
                return;
            }
            if (std::find(candidates.begin(), candidates.begin() + count, candidate) == candidates.begin() + count) {
                candidates[count++] = candidate;
            }
        };

        append_candidate(cell_id);
        for (uint32_t neighbor : topology.cells[cell_id].neighbor_indices) {
            append_candidate(neighbor);
        }

        resolver.candidates_by_cell[cell_id] = candidates;
        resolver.candidate_counts[cell_id] = static_cast<uint8_t>(count);
    }

    return resolver;
}

float resolver_dot(const PointOwnershipResolver& resolver, uint32_t cell_id, Vec3 position) {
    return position.x * resolver.center_x[cell_id] +
           position.y * resolver.center_y[cell_id] +
           position.z * resolver.center_z[cell_id];
}

uint32_t nearest_local_cell_on_sphere(const PointOwnershipResolver& resolver, uint32_t source_cell_id, Vec3 normalized_position) {
    if (source_cell_id >= resolver.candidates_by_cell.size()) {
        return 0;
    }

    const auto& candidates = resolver.candidates_by_cell[source_cell_id];
    const uint32_t count = resolver.candidate_counts[source_cell_id];
    uint32_t nearest = source_cell_id;
    float best_dot = -2.0f;

#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
    const __m128 px = _mm_set1_ps(normalized_position.x);
    const __m128 py = _mm_set1_ps(normalized_position.y);
    const __m128 pz = _mm_set1_ps(normalized_position.z);
    alignas(16) float scores[4] = {};
    uint32_t i = 0;
    for (; i + 3u < count; i += 4u) {
        const uint32_t c0 = candidates[i];
        const uint32_t c1 = candidates[i + 1u];
        const uint32_t c2 = candidates[i + 2u];
        const uint32_t c3 = candidates[i + 3u];
        const __m128 xs = _mm_set_ps(resolver.center_x[c3], resolver.center_x[c2], resolver.center_x[c1], resolver.center_x[c0]);
        const __m128 ys = _mm_set_ps(resolver.center_y[c3], resolver.center_y[c2], resolver.center_y[c1], resolver.center_y[c0]);
        const __m128 zs = _mm_set_ps(resolver.center_z[c3], resolver.center_z[c2], resolver.center_z[c1], resolver.center_z[c0]);
        const __m128 score = _mm_add_ps(_mm_add_ps(_mm_mul_ps(px, xs), _mm_mul_ps(py, ys)), _mm_mul_ps(pz, zs));
        _mm_store_ps(scores, score);
        for (uint32_t lane = 0; lane < 4u; ++lane) {
            if (scores[lane] > best_dot) {
                best_dot = scores[lane];
                nearest = candidates[i + lane];
            }
        }
    }
    for (; i < count; ++i) {
        const uint32_t candidate = candidates[i];
        const float score = resolver_dot(resolver, candidate, normalized_position);
        if (score > best_dot) {
            best_dot = score;
            nearest = candidate;
        }
    }
#else
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t candidate = candidates[i];
        const float score = resolver_dot(resolver, candidate, normalized_position);
        if (score > best_dot) {
            best_dot = score;
            nearest = candidate;
        }
    }
#endif

    return nearest;
}

void append_sample(PointCloud& points, Vec3 position, uint32_t source_cell_id, uint32_t owner_cell_id, uint32_t material_id) {
    const Vec3 normalized_position = normalize(position);
    points.positions.push_back(normalized_position);
    points.normals.push_back(normalized_position);
    points.source_cell_ids.push_back(source_cell_id);
    points.owner_cell_ids.push_back(owner_cell_id);
    points.material_ids.push_back(material_id);
}

void build_owned_radius_lookup(PointCloud& points, uint32_t cell_count) {
    points.owned_radius_by_cell.assign(cell_count, 1.0f);
}

} // namespace

PointCloud build_surface_point_cloud(const GoldbergTopology& topology) {
    PointCloud points;
    const size_t sample_count = topology.cells.size() * 4;
    points.positions.reserve(sample_count);
    points.normals.reserve(sample_count);
    points.source_cell_ids.reserve(sample_count);
    points.owner_cell_ids.reserve(sample_count);
    points.material_ids.reserve(sample_count);
    const PointOwnershipResolver ownership = build_ownership_resolver(topology);

    for (uint32_t cell_id = 0; cell_id < topology.cells.size(); ++cell_id) {
        const GoldbergCell& cell = topology.cells[cell_id];
        const uint32_t material = cell.kind == GoldbergCellKind::Pentagon ? 1u : 0u;

        append_sample(points, cell.center, cell_id, cell_id, material);

        const uint32_t corner_count = static_cast<uint32_t>(cell.corner_indices.size());
        const std::array<uint32_t, 3> sample_corners = {
            0u,
            corner_count / 3u,
            (corner_count * 2u) / 3u,
        };

        for (uint32_t corner_slot : sample_corners) {
            const uint32_t corner_index = cell.corner_indices[corner_slot % corner_count];
            const Vec3 corner = topology.vertices[corner_index].position;
            const Vec3 position = normalize(lerp(cell.center, corner, 0.38f));
            append_sample(points, position, cell_id, nearest_local_cell_on_sphere(ownership, cell_id, position), material + 2u);
        }
    }

    build_owned_radius_lookup(points, static_cast<uint32_t>(topology.cells.size()));
    return points;
}

PointCloudValidation validate_point_cloud(const GoldbergTopology& topology, const PointCloud& points) {
    std::ostringstream report;
    bool ok = true;
    uint32_t mismatch_count = 0;

    auto require = [&](bool condition, const char* message) {
        if (!condition) {
            ok = false;
            report << "Point assignment failed: " << message << '\n';
        }
    };

    require(points.size() == topology.cells.size() * 4, "point count is not cell_count * 4");
    require(points.normals.size() == points.size(), "normal count does not match point count");
    require(points.source_cell_ids.size() == points.size(), "source cell id count does not match point count");
    require(points.owner_cell_ids.size() == points.size(), "owner cell id count does not match point count");
    require(points.material_ids.size() == points.size(), "material id count does not match point count");
    require(points.owned_radius_by_cell.size() == topology.cells.size(), "owned radius lookup count does not match cell count");

    for (uint32_t i = 0; i < points.size(); ++i) {
        require(points.source_cell_ids[i] < topology.cells.size(), "source cell id is out of range");
        require(points.owner_cell_ids[i] < topology.cells.size(), "owner cell id is out of range");
        require(std::fabs(length(points.positions[i]) - 1.0f) < 0.0005f, "point position is not on the unit sphere");

        if (points.source_cell_ids[i] != points.owner_cell_ids[i]) {
            ++mismatch_count;
        }
    }

    if (ok) {
        report << "Point assignment OK: " << points.size() << " samples, "
               << mismatch_count << " ownership mismatches.";
    }

    return {ok, mismatch_count, report.str()};
}

} // namespace ae
