#pragma once

#include <memory>

namespace telef::solver {

/**
 * Pluggable local parameterization (manifold) for a parameter block.
 * Like Ceres: the solver is task-invariant and calls Plus() when applying updates.
 * Implementations (Euclidean, SO(3), quaternion, etc.) live outside the core solver.
 */
class LocalParameterization {
public:
    using Ptr = std::shared_ptr<LocalParameterization>;
    virtual ~LocalParameterization() = default;

    /**
     * Apply delta in tangent space to state: state_plus_delta = Plus(state, delta).
     * All pointers are host memory; size is the parameter block size.
     */
    virtual void Plus(const float* state, const float* delta, float* state_plus_delta, int size) const = 0;
};

/** Euclidean: state_plus_delta[i] = state[i] + delta[i]. */
class EuclideanParameterization : public LocalParameterization {
public:
    void Plus(const float* state, const float* delta, float* state_plus_delta, int size) const override {
        for (int i = 0; i < size; ++i) state_plus_delta[i] = state[i] + delta[i];
    }
};

}  // namespace telef::solver
