#ifndef HOPS_GAMMAMODEL_HPP
#define HOPS_GAMMAMODEL_HPP

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>
#include <utility>

#include "Model.hpp"

namespace hops {
    /**
     * @brief Reference http://www.stats.org.uk/priors/noninformative/YangBerger1998.pdf.
     */
    class Gamma : public Model {
    public:

        /**
         * @brief constructor for single measurement. The Gamma will have dimensionality 2*measurement.rows(), because
         * every row of the measurement is distributed according to a gamma distribution, which has 2 parameters.
         * @param measurements
         */
        explicit Gamma(const Eigen::VectorXd &measurement) : Gamma(std::vector{measurement}) {}

        /**
         * @brief constructor for set of measurements. The Gamma will have dimensionality 2*measurement.rows(), because
         * every row of the measurement is distributed according to a gamma distribution, which has 2 parameters.
         * @param measurements each of the measurements is expected to have the same size. Should not be empty!
         */
        explicit Gamma(std::vector<Eigen::VectorXd> measurements) : measurements(std::move(measurements)) {
            long measurement_dim = this->measurements[0].rows();
            measurements_sum = std::accumulate(this->measurements.begin(),
                                               this->measurements.end(),
                                               Eigen::VectorXd(Eigen::VectorXd::Zero(measurement_dim)),
                                               [](const Eigen::VectorXd &m1, const Eigen::VectorXd &m2) {
                                                   return Eigen::VectorXd(m1 + m2);
                                               });

            log_measurements_sum = std::accumulate(this->measurements.begin(),
                                                   this->measurements.end(),
                                                   Eigen::VectorXd(Eigen::VectorXd::Ones(measurement_dim)),
                                                   [](const Eigen::VectorXd &m1, const Eigen::VectorXd &m2) {
                                                       return Eigen::VectorXd(
                                                               m1.array().log().matrix() + m2.array().log().matrix());
                                                   }
            );
            if (this->measurements.empty()) {
                throw std::invalid_argument("measurements vector should not be empty (Gamma).");
            }
        }

        [[nodiscard]] std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) const override {
            VectorType gradient = Eigen::VectorXd::Zero(x.rows());
            for (long i = 0; i < measurements[0].rows(); ++ ++i) {
                // reference https://en.wikipedia.org/wiki/Gamma_distribution
                double k = x(2 * i);
                double theta = x(2 * i + 1);
                auto N = static_cast<double>(measurements.size());
                // digamma is derivative of ln(gamma)
                gradient(2 * i) = log_measurements_sum(i) - N * std::log(theta) - N * boost::math::digamma(k);
                gradient(2 * i + 1) = measurements_sum(i) / std::pow(theta, 2) - N * k / theta;
            }

            return gradient;
        }

        [[nodiscard]] std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &x) const override {
            MatrixType fisherInformation = Eigen::MatrixXd::Zero(x.rows(), x.rows());
            for (long i = 0; i < measurements[0].rows(); ++ ++i) {
                // reference http://www.stats.org.uk/priors/noninformative/YangBerger1998.pdf
                double k = x(2 * i);
                double theta = x(2 * i + 1);
                auto N = static_cast<double>(measurements.size());

                fisherInformation(2 * i, 2 * i) = boost::math::polygamma(1, k);
                fisherInformation(2 * i + 1, 2 * i) = 1. / theta;
                fisherInformation(2 * i, 2 * i + 1) = fisherInformation(2 * i + 1, 2 * i);
                fisherInformation(2 * i + 1, 2 * i + 1) = k / std::pow(theta, 2);
            }

            return fisherInformation;
        }

        [[nodiscard]] std::optional<std::vector<std::string>> getParameterNames() const override {
            std::vector<std::string> parameterNames;
            for (long i = 0; i < measurements[0].rows(); ++i) {
                parameterNames.emplace_back("shape " + std::to_string(i));
                parameterNames.emplace_back("scale " + std::to_string(i));
            }
            return parameterNames;
        }

        [[nodiscard]] double computeNegativeLogLikelihood(const VectorType &x) const override {
            double l = 0;
            for (long i = 0; i < measurements[0].rows(); ++i) {
                // reference https://en.wikipedia.org/wiki/Gamma_distribution
                double k = x(2 * i);
                double theta = x(2 * i + 1);
                auto N = static_cast<double>(measurements.size());

                l += (k - 1) * log_measurements_sum(i)
                     - measurements_sum(i) / theta
                     - N * k * std::log(theta)
                     - N * std::log(std::tgamma(k));
            }
            return l;
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<Gamma>(*this);
        }

    private:
        std::vector<Eigen::VectorXd> measurements;
        Eigen::VectorXd measurements_sum;
        Eigen::VectorXd log_measurements_sum;
    };
}

#endif //HOPS_GAMMAMODEL_HPP
