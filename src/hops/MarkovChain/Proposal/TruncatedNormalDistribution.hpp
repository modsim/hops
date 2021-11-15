#ifndef HOPS_TRUNCATEDNORMALDISTRIBUTION_HPP
#define HOPS_TRUNCATEDNORMALDISTRIBUTION_HPP

#include <random>

namespace hops {

    /**
     * @brief Truncated normal distribution with mean 0.
     * @tparam RealType
     */
    template<typename RealType>
    class TruncatedNormalDistribution {
    public:
        struct param_type {
            RealType sigma;
            RealType lowerBound = -std::numeric_limits<RealType>::infinity();
            RealType upperBound = std::numeric_limits<RealType>::infinity();
            RealType phiLower;
            RealType phiUpper;

            void setPhi() {
                if (lowerBound != -std::numeric_limits<RealType>::infinity())
                    phiLower = Phi(lowerBound / sigma);
                else
                    phiLower = 0;

                if (upperBound != std::numeric_limits<RealType>::infinity())
                    phiUpper = Phi(upperBound / sigma);
                else
                    phiUpper = 1;
            }

            param_type(RealType sigma, RealType lowerBound, RealType upperBound) :
                    sigma(sigma), lowerBound(lowerBound), upperBound(upperBound) {
                setPhi();
            }
        };

        template<typename Generator>
        RealType operator()(Generator &g, const param_type &params) {
            RealType uniformNumber = uniformRealDistribution.operator()(g);
            return inverseCumulativeDensityFunction(uniformNumber, params);
        }

        RealType inverseNormalization(const param_type &params) {
            return params.phiUpper - params.phiLower;
        }

        RealType probabilityDensity(RealType x, RealType sigma, RealType lowerBound, RealType upperBound){
            RealType pdf = 1./(sigma * sqrt_2pi) * std::exp(-(1./2)*std::pow(x/sigma, 2));
            return pdf / inverseNormalization(param_type(sigma, lowerBound, upperBound));
        }

    private:
        RealType inverseCumulativeDensityFunction(RealType x, const param_type &params) const {
            x *= params.phiUpper - params.phiLower;
            x += params.phiLower;
            return inv_Phi(x) * params.sigma;
        }

        std::uniform_real_distribution<> uniformRealDistribution{0, 1};

        static const constexpr RealType one_over_sqrt_2pi = RealType(0.398942280401432677939946);
        static const constexpr RealType sqrt_2pi = RealType(2.50662827463100050241577);
        static const constexpr RealType one_over_sqrt_2 = RealType(0.707106781186547524400845);

        //	following code is adapted from https://github.com/rabauke/trng4

        static RealType Phi(RealType x) {
            return 0.5 + 0.5 * std::erf(one_over_sqrt_2 * x);
        }

        // this function is based on an approximation by Peter J. Acklam
        // see http://home.online.no/~pjacklam/notes/invnorm/ for details

        struct inv_Phi_traits {
            static RealType a(int i) throw() {
                const RealType a_[] = {
                        -3.969683028665376e+01, 2.209460984245205e+02,
                        -2.759285104469687e+02, 1.383577518672690e+02,
                        -3.066479806614716e+01, 2.506628277459239e+00};
                return a_[i];
            }

            static RealType b(int i) throw() {
                const RealType b_[] = {
                        -5.447609879822406e+01, 1.615858368580409e+02,
                        -1.556989798598866e+02, 6.680131188771972e+01,
                        -1.328068155288572e+01};
                return b_[i];
            }

            static RealType c(int i) throw() {
                const RealType c_[] = {
                        -7.784894002430293e-03, -3.223964580411365e-01,
                        -2.400758277161838e+00, -2.549732539343734e+00,
                        4.374664141464968e+00, 2.938163982698783e+00};
                return c_[i];
            }

            static RealType d(int i) throw() {
                const RealType d_[] = {
                        7.784695709041462e-03, 3.224671290700398e-01,
                        2.445134137142996e+00, 3.754408661907416e+00};
                return d_[i];
            }

            static RealType x_low() throw() { return 0.02425; }

            static RealType x_high() throw() { return 1.0 - 0.02425; }

            static RealType zero() throw() { return 0.0; }

            static RealType one() throw() { return 1.0; }

            static RealType one_half() throw() { return 0.5; }

            static RealType minus_two() throw() { return -2.0; }
        };

        static RealType inv_Phi(RealType x) {
            if (x < inv_Phi_traits::zero() || x > inv_Phi_traits::one())
                return std::numeric_limits<RealType>::quiet_NaN();
            if (x == inv_Phi_traits::zero())
                return -std::numeric_limits<RealType>::infinity();
            if (x == inv_Phi_traits::one())
                return std::numeric_limits<RealType>::infinity();

            RealType t, q;
            if (x < inv_Phi_traits::x_low()) {
                // Rational approximation for lower region
                q = std::sqrt(inv_Phi_traits::minus_two() * std::log(x));
                t = (((((inv_Phi_traits::c(0) * q + inv_Phi_traits::c(1)) * q +
                        inv_Phi_traits::c(2)) * q + inv_Phi_traits::c(3)) * q +
                      inv_Phi_traits::c(4)) * q + inv_Phi_traits::c(5)) /
                    ((((inv_Phi_traits::d(0) * q + inv_Phi_traits::d(1)) * q +
                       inv_Phi_traits::d(2)) * q + inv_Phi_traits::d(3)) * q +
                     inv_Phi_traits::one());
            } else if (x < inv_Phi_traits::x_high()) {
                // Rational approximation for central region
                q = x - inv_Phi_traits::one_half();
                RealType r = q * q;
                t = (((((inv_Phi_traits::a(0) * r + inv_Phi_traits::a(1)) * r +
                        inv_Phi_traits::a(2)) * r + inv_Phi_traits::a(3)) * r +
                      inv_Phi_traits::a(4)) * r + inv_Phi_traits::a(5)) * q /
                    (((((inv_Phi_traits::b(0) * r + inv_Phi_traits::b(1)) * r +
                        inv_Phi_traits::b(2)) * r + inv_Phi_traits::b(3)) * r +
                      inv_Phi_traits::b(4)) * r + inv_Phi_traits::one());
            } else {
                // Rational approximation for upper region
                q = std::sqrt(inv_Phi_traits::minus_two() * std::log(1.0 - x));
                t = -(((((inv_Phi_traits::c(0) * q + inv_Phi_traits::c(1)) * q +
                         inv_Phi_traits::c(2)) * q + inv_Phi_traits::c(3)) * q +
                       inv_Phi_traits::c(4)) * q + inv_Phi_traits::c(5)) /
                    ((((inv_Phi_traits::d(0) * q + inv_Phi_traits::d(1)) * q +
                       inv_Phi_traits::d(2)) * q + inv_Phi_traits::d(3)) * q +
                     inv_Phi_traits::one());
            }

            // refinement by Halley rational method
            if (std::numeric_limits<RealType>::epsilon() < 1e-9) {
                RealType e(Phi(t) - x);
                RealType u(e * sqrt_2pi * std::exp(t * t * inv_Phi_traits::one_half()));
                t -= u / (inv_Phi_traits::one() + t * u * inv_Phi_traits::one_half());
            }
            return t;
        }
    };
}

#endif //HOPS_TRUNCATEDNORMALDISTRIBUTION_HPP
