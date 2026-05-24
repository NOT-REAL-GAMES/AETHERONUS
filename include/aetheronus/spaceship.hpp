#pragma once

#include "aetheronus/math.hpp"

#include <vector>

namespace ae {

struct SpaceshipInput {
    float yaw = 0.0f;
    float pitch = 0.0f;
    float roll = 0.0f;
    float throttle = 0.0f;
};

struct SpaceshipState {
    Vec3 position = {0.0f, 0.25f, 2.9f};
    Vec3 forward = {0.0f, 0.0f, -1.0f};
    Vec3 up = {0.0f, 1.0f, 0.0f};
    float throttle = 0.0f;
    float yaw_velocity = 0.0f;
    float pitch_velocity = 0.0f;
    float roll_velocity = 0.0f;
    std::vector<Vec3> trail;
};

Vec3 spaceship_position(const SpaceshipState& ship);
void update_spaceship(SpaceshipState& ship, const SpaceshipInput& input, float dt);

} // namespace ae
