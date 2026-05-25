#include "aetheronus/spaceship.hpp"

#include <algorithm>
#include <cmath>

namespace ae {
namespace {

Vec3 safe_perpendicular(Vec3 direction) {
    const Vec3 reference = std::fabs(direction.y) < 0.92f ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
    return normalize(cross(direction, reference));
}

Vec3 rotate_around_axis(Vec3 value, Vec3 axis, float radians) {
    axis = normalize(axis);
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return value * c + cross(axis, value) * s + axis * (dot(axis, value) * (1.0f - c));
}

void orthonormalize_ship_basis(SpaceshipState& ship) {
    ship.forward = normalize(ship.forward);
    ship.up = ship.up - ship.forward * dot(ship.up, ship.forward);
    if (length(ship.up) <= 0.00001f) {
        ship.up = safe_perpendicular(ship.forward);
    } else {
        ship.up = normalize(ship.up);
    }
}

} // namespace

Vec3 spaceship_position(const SpaceshipState& ship) {
    return ship.position;
}

void update_spaceship(SpaceshipState& ship, const SpaceshipInput& input, float dt) {
    const float clamped_dt = std::clamp(dt, 0.0f, 0.05f);
    orthonormalize_ship_basis(ship);

    ship.throttle = std::clamp(ship.throttle + input.throttle * clamped_dt * 0.55f, 0.0f, 1.0f);

    const float damping = std::exp(-clamped_dt * 3.8f);
    ship.yaw_velocity = ship.yaw_velocity * damping + input.yaw * clamped_dt * 18.0f;
    ship.pitch_velocity = ship.pitch_velocity * damping + input.pitch * clamped_dt * 18.0f;
    const float roll_target = std::clamp(input.roll, -1.0f, 1.0f) * 2.6f;
    ship.roll_velocity += (roll_target - ship.roll_velocity) * std::clamp(clamped_dt * 7.5f, 0.0f, 1.0f);

    const Vec3 right = normalize(cross(ship.forward, ship.up));
    ship.forward = rotate_around_axis(ship.forward, ship.up, ship.yaw_velocity * clamped_dt);
    ship.forward = rotate_around_axis(ship.forward, right, ship.pitch_velocity * clamped_dt);
    ship.up = rotate_around_axis(ship.up, right, ship.pitch_velocity * clamped_dt);
    ship.up = rotate_around_axis(ship.up, ship.forward, ship.roll_velocity * clamped_dt);
    orthonormalize_ship_basis(ship);

    const float speed = ship.throttle * 80.0f;
    ship.position = ship.position + ship.forward * (speed * clamped_dt);

    const uint32_t newest_index = (ship.trail_head + SpaceshipState::TrailCapacity - 1u) % SpaceshipState::TrailCapacity;
    if (ship.trail_count == 0 || length(ship.position - ship.trail[newest_index]) > 10.0f) {
        ship.trail[ship.trail_head] = ship.position;
        ship.trail_head = (ship.trail_head + 1u) % SpaceshipState::TrailCapacity;
        ship.trail_count = std::min(ship.trail_count + 1u, SpaceshipState::TrailCapacity);
    }
}

} // namespace ae
