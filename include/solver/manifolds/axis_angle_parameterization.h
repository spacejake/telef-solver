#pragma once

#include <cmath>
#include "solver/localParameterization.h"

namespace telef::solver {

/**
 * SO(3) axis-angle local parameterization: Plus(u, delta) gives u_new such that
 * R(u_new) = R(u) * exp(delta). Lives outside the core solver; link only when needed.
 */
class AxisAngleParameterization : public LocalParameterization {
public:
    void Plus(const float* state, const float* delta, float* state_plus_delta, int size) const override {
        if (size != 3) {
            for (int i = 0; i < size; ++i) state_plus_delta[i] = state[i] + delta[i];
            return;
        }
        float Ru[9], Rd[9], Rnew[9];
        rodrigues(delta, Rd);
        axis_angle_to_R(state, Ru);
        mat3_mul(Ru, Rd, Rnew);
        R_to_axis_angle(Rnew, state_plus_delta);
    }

private:
    static void rodrigues(const float* omega, float* R) {
        float theta_sq = omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2];
        float theta = std::sqrt(theta_sq);
        const float eps = 1e-8f;
        if (theta < eps) {
            R[0] = 1.f;  R[1] = -omega[2]; R[2] = omega[1];
            R[3] = omega[2];  R[4] = 1.f;  R[5] = -omega[0];
            R[6] = -omega[1]; R[7] = omega[0]; R[8] = 1.f;
            return;
        }
        float c = std::cos(theta), s = std::sin(theta);
        float ax = omega[0]/theta, ay = omega[1]/theta, az = omega[2]/theta;
        float one_minus_c = 1.f - c;
        R[0] = c + ax*ax*one_minus_c;
        R[1] = ax*ay*one_minus_c - az*s;
        R[2] = ax*az*one_minus_c + ay*s;
        R[3] = ax*ay*one_minus_c + az*s;
        R[4] = c + ay*ay*one_minus_c;
        R[5] = ay*az*one_minus_c - ax*s;
        R[6] = ax*az*one_minus_c - ay*s;
        R[7] = ay*az*one_minus_c + ax*s;
        R[8] = c + az*az*one_minus_c;
    }

    static void R_to_axis_angle(const float* R, float* u) {
        float trace = R[0] + R[4] + R[8];
        float theta = std::acos(std::max(-1.f, std::min(1.f, (trace - 1.f) * 0.5f)));
        const float eps = 1e-6f;
        if (theta < eps) {
            u[0] = u[1] = u[2] = 0.f;
            return;
        }
        float sin_theta = std::sin(theta);
        if (std::fabs(sin_theta) < eps) {
            u[0] = u[1] = u[2] = 0.f;
            return;
        }
        float k = theta / sin_theta;
        u[0] = k * (R[7] - R[5]);
        u[1] = k * (R[2] - R[6]);
        u[2] = k * (R[3] - R[1]);
    }

    static void axis_angle_to_R(const float* u, float* R) {
        rodrigues(u, R);
    }

    static void mat3_mul(const float* A, const float* B, float* C) {
        C[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
        C[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
        C[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];
        C[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
        C[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
        C[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];
        C[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
        C[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
        C[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
    }
};

}  // namespace telef::solver
