#ifndef HOPS_MIXTURE_HPP
#define HOPS_MIXTURE_HPP

#include <numeric>
#include <utility>
#include <vector>

#include "hops/Model/Model.hpp"

namespace hops {
    class Mixture : public Model {
    public:
        explicit Mixture(const std::vector<std::shared_ptr<Model>> &components) :
                components(components),
                weights(std::vector<double>(components.size(), 1. / components.size())) {}

        Mixture(std::vector<std::shared_ptr<Model>> components, std::vector<double> weights) :
                components(std::move(components)),
                weights(std::move(weights)) {
            if (this->components.size() != this->weights.size()) {
                throw std::invalid_argument(
                        "Components and weights should have the same length for construction of mixture model."
                );
            }
        }

        /**
         * @brief Implementation using the logsumexp trick.
         * @param x evaluation point
         * @return log density value
         */
        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override {
            std::vector<double> negLogLikelihoods;
            for (const auto &component: components) {
                negLogLikelihoods.push_back(component->computeNegativeLogLikelihood(x));
            }

            auto minNegLogLikeIt = std::min_element(negLogLikelihoods.begin(), negLogLikelihoods.end());
            double likelihoodMinusMax = std::transform_reduce(components.begin(),
                                                              components.end(),
                                                              weights.begin(),
                                                              double(0.),
                                                              std::plus<>(),
                                                              [&](const std::shared_ptr<Model> &model, double weight) {
                                                                  return static_cast<double>(weight *
                                                                                             std::exp(*minNegLogLikeIt
                                                                                                      -
                                                                                                      model->computeNegativeLogLikelihood(
                                                                                                              x)));
                                                              });
            return *minNegLogLikeIt - std::log(likelihoodMinusMax);
        }

        /**
         * @brief Implementation by using the logsumexp trick and the chain rule.
         * @param x
         * @return
         */
        [[nodiscard]] std::optional<VectorType>
        computeLogLikelihoodGradient(const VectorType &x) override {
            std::vector<double> negLogLikelihoods;
            for (const auto &component: components) {
                negLogLikelihoods.push_back(component->computeNegativeLogLikelihood(x));
            }
            auto minNegLogLikeIt = std::min_element(negLogLikelihoods.begin(), negLogLikelihoods.end());

            std::vector<double> weightedLikelihoods;
            std::transform(negLogLikelihoods.begin(),
                           negLogLikelihoods.end(),
                           weights.begin(),
                           std::back_inserter(weightedLikelihoods),
                           [&minNegLogLikeIt](double negLogLikelihood, double weight) {
                               return weight * std::exp(*minNegLogLikeIt - negLogLikelihood);
                           }
            );

            double denominator = std::accumulate(weightedLikelihoods.begin(), weightedLikelihoods.end(), 0.);

            std::vector<VectorType> logLikelihoodGradients;
            for (const auto &model: components) {
                auto gradient = model->computeLogLikelihoodGradient(x);
                if (gradient) {
                    logLikelihoodGradients.emplace_back(gradient.value());
                } else {
                    logLikelihoodGradients.emplace_back(VectorType::Zero(x.rows()));
                }
            }
            const auto &maxLogLikelihoodGradient = logLikelihoodGradients[std::distance(negLogLikelihoods.begin(),
                                                                                        minNegLogLikeIt)];

            return maxLogLikelihoodGradient + std::transform_reduce(logLikelihoodGradients.begin(),
                                                                    logLikelihoodGradients.end(),
                                                                    weightedLikelihoods.begin(),
                                                                    VectorType(VectorType::Zero(x.rows())),
                                                                    std::plus<>(),
                                                                    [&](const VectorType &logLikelihoodGradient,
                                                                        double weightedLikelihood) {
                                                                        return weightedLikelihood *
                                                                               (logLikelihoodGradient -
                                                                                maxLogLikelihoodGradient);
                                                                    }) / denominator;
        }

        [[nodiscard]] std::optional<MatrixType>
        computeExpectedFisherInformation(const VectorType &) override {
            return std::nullopt;
        }

        [[nodiscard]] const std::vector<std::shared_ptr<Model>> &getComponents() const {
            return components;
        }

        [[nodiscard]] const std::vector<double> &getWeights() const {
            return weights;
        }

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override {
            return components.front()->getDimensionNames();
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<Mixture>(components, weights);
        }

    private:
        std::vector<std::shared_ptr<Model>> components;
        std::vector<double> weights;
    };
}

#endif //HOPS_MIXTURE_HPP
