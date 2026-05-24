#include "aetheronus/math.hpp"

#include <algorithm>

namespace ae {

Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator-(Vec3 v) {
    return {-v.x, -v.y, -v.z};
}

Vec3 operator*(Vec3 v, float scalar) {
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

Vec3 operator*(float scalar, Vec3 v) {
    return v * scalar;
}

Vec3 operator/(Vec3 v, float scalar) {
    return {v.x / scalar, v.y / scalar, v.z / scalar};
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length(Vec3 v) {
    return std::sqrt(dot(v, v));
}

Vec3 normalize(Vec3 v) {
    const float len = length(v);
    if (len <= 0.000001f) {
        return {0.0f, 1.0f, 0.0f};
    }
    return v / len;
}

Vec3 lerp(Vec3 a, Vec3 b, float t) {
    return a * (1.0f - t) + b * t;
}

Mat4 identity() {
    Mat4 result;
    result.m[0] = 1.0f;
    result.m[5] = 1.0f;
    result.m[10] = 1.0f;
    result.m[15] = 1.0f;
    return result;
}

Mat4 perspective(float fov_y_radians, float aspect, float near_plane, float far_plane) {
    Mat4 result;
    const float f = 1.0f / std::tan(fov_y_radians * 0.5f);
    result.m[0] = f / aspect;
    result.m[5] = f;
    result.m[10] = (far_plane + near_plane) / (near_plane - far_plane);
    result.m[11] = -1.0f;
    result.m[14] = (2.0f * far_plane * near_plane) / (near_plane - far_plane);
    return result;
}

Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    Mat4 result = identity();
    result.m[0] = s.x;
    result.m[4] = s.y;
    result.m[8] = s.z;
    result.m[1] = u.x;
    result.m[5] = u.y;
    result.m[9] = u.z;
    result.m[2] = -f.x;
    result.m[6] = -f.y;
    result.m[10] = -f.z;
    result.m[12] = -dot(s, eye);
    result.m[13] = -dot(u, eye);
    result.m[14] = dot(f, eye);
    return result;
}

Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result.m[column * 4 + row] =
                a.m[0 * 4 + row] * b.m[column * 4 + 0] +
                a.m[1 * 4 + row] * b.m[column * 4 + 1] +
                a.m[2 * 4 + row] * b.m[column * 4 + 2] +
                a.m[3 * 4 + row] * b.m[column * 4 + 3];
        }
    }
    return result;
}

} // namespace ae
