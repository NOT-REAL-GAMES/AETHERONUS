#pragma once

#include "aetheronus/math.hpp"
#include "aetheronus/planet_scale.hpp"

#include <array>
#include <cstdint>

namespace ae {

struct SpaceshipInput {
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    float throttle = 0.0f;
};

struct SpaceshipState {
    Vec3 position = {0.0f, PlanetRadiusKilometers + 250.0f, 0.0f};
    Vec3 forward = {1.0f, 0.0f, 0.0f};
    Vec3 up = {0.0f, 1.0f, 0.0f};
    float throttle = 0.0f;
    float yaw_velocity = 0.0f;
    float pitch_velocity = 0.0f;
    float roll_velocity = 0.0f;
    static constexpr uint32_t TrailCapacity = 64;
    std::array<Vec3, TrailCapacity> trail = {};
    uint32_t trail_head = 0;
    uint32_t trail_count = 0;
};

Vec3 spaceship_position(const SpaceshipState& ship);
void update_spaceship(SpaceshipState& ship, const SpaceshipInput& input, float dt);

} // namespace ae
