#pragma once

#include "aetheronus/math.hpp"

namespace ae {

constexpr float PlanetCircumferenceKilometers = 40000.0f;
constexpr float PlanetRadiusKilometers = PlanetCircumferenceKilometers / (2.0f * Pi);
constexpr float PlanetRadiusWorldUnits = 1.0f;
constexpr float KilometersPerWorldUnit = PlanetRadiusKilometers / PlanetRadiusWorldUnits;

constexpr float world_units_to_kilometers(float units) {
    return units * KilometersPerWorldUnit;
}

constexpr float kilometers_to_world_units(float kilometers) {
    return kilometers / KilometersPerWorldUnit;
}

} // namespace ae
