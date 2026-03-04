#pragma once

#include <memory>
#include <cmath>

namespace telef::solver {
    /**
     * Robust loss for outlier-resistant optimization.
     * Cost is 0.5 * rho(r^2); gradient of cost uses rho'(r^2) * r * J.
     * We scale residuals and Jacobian rows by sqrt(rho'(r^2)) so existing J'J and J'r stay correct.
     */
    class LossFunction {
    public:
        using Ptr = std::shared_ptr<LossFunction>;
        virtual ~LossFunction() = default;
        /** rho'(r^2); for scaling use sqrt(rho_prime). */
        virtual float rhoPrime(float r_sq) const = 0;
    };

    /** Huber: linear beyond delta. rho'(r^2) = 1 if r^2 <= delta^2 else delta/sqrt(r^2). */
    class HuberLoss : public LossFunction {
    public:
        explicit HuberLoss(float delta) : delta_(delta), delta_sq_(delta * delta) {}
        float rhoPrime(float r_sq) const override {
            if (r_sq <= delta_sq_) return 1.f;
            float r = std::sqrt(r_sq);
            return r > 1e-10f ? delta_ / r : 1.f;
        }
        float getDelta() const { return delta_; }
    private:
        float delta_;
        float delta_sq_;
    };
}
