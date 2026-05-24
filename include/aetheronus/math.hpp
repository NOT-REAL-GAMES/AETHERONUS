#pragma once

#include <cmath>

namespace ae {

constexpr float Pi = 3.14159265358979323846f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Mat4 {
    float m[16] = {};
};

Vec3 operator+(Vec3 a, Vec3 b);
Vec3 operator-(Vec3 a, Vec3 b);
Vec3 operator-(Vec3 v);
Vec3 operator*(Vec3 v, float scalar);
Vec3 operator*(float scalar, Vec3 v);
Vec3 operator/(Vec3 v, float scalar);

float dot(Vec3 a, Vec3 b);
Vec3 cross(Vec3 a, Vec3 b);
float length(Vec3 v);
Vec3 normalize(Vec3 v);
Vec3 lerp(Vec3 a, Vec3 b, float t);

Mat4 identity();
Mat4 perspective(float fov_y_radians, float aspect, float near_plane, float far_plane);
Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up);
Mat4 operator*(const Mat4& a, const Mat4& b);

} // namespace ae
