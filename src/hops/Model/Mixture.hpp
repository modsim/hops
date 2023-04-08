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

        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) override {
            double likelihood = std::transform_reduce(components.begin(),
                                                      components.end(),
                                                      weights.begin(),
                                                      double(0.),
                                                      std::plus<>(),
                                                      [&x](const std::shared_ptr<Model> &model, double weight) {
                                                          return static_cast<double>(weight * std::exp(
                                                                  -model->computeNegativeLogLikelihood(x)));
                                                      });
            return -std::log(likelihood);
        }

        /**
         * @ Brief Implementation derived from using chain rule on log(sum(...)).
         * @param x
         * @return
         */
        [[nodiscard]] std::optional<VectorType>
        computeLogLikelihoodGradient(const VectorType &x) override {
            std::vector<double> weightedLikelihoods;

            std::transform(components.begin(),
                           components.end(),
                           weights.begin(),
                           std::back_inserter(weightedLikelihoods),
                           [&x](const std::shared_ptr<Model> &model, double weight) {
                               return weight * std::exp(-model->computeNegativeLogLikelihood(x));
                           }
            );

            double denominator = std::accumulate(weightedLikelihoods.begin(), weightedLikelihoods.end(), 0.);

            return std::transform_reduce(components.begin(),
                               components.end(),
                               weightedLikelihoods.begin(),
                               VectorType(VectorType::Zero(x.rows())),
                               std::plus<>(),
                               [&x](const std::shared_ptr<Model> &model, double weightedLikelihood) {
                                   auto gradient = model->computeLogLikelihoodGradient(x);
                                   if (gradient) {
                                       return VectorType(weightedLikelihood * gradient.value());
                                   }
                                   return VectorType(VectorType::Zero(x.rows()));
                               }
            ) / denominator;
        }

        [[nodiscard]] std::optional<MatrixType>
        computeExpectedFisherInformation(const VectorType &) override {
            return std::nullopt;
        }

        [[nodiscard]] const std::vector<std::shared_ptr<Model>>& getComponents() const {
            return components;
        }

        [[nodiscard]] const std::vector<double>& getWeights() const {
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
