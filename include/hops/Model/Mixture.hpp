#ifndef HOPS_MIXTURE_HPP
#define HOPS_MIXTURE_HPP

#include <execution>
#include <numeric>
#include <utility>
#include <vector>

#include <hops/Model/Model.hpp>

namespace hops {
    class Mixture : public Model {
    public:
        explicit Mixture(const std::vector<std::shared_ptr<Model>> &models) :
                models(models),
                weights(std::vector<double>(models.size(), 1. / models.size())) {}

        Mixture(std::vector<std::shared_ptr<Model>> models, std::vector<double> weights) :
                models(std::move(models)),
                weights(std::move(weights)) {
            if (this->models.size() != this->weights.size()) {
                throw std::invalid_argument(
                        "Models and weights should have the same length for construction of Multimodal Object."
                );
            }
        }

        [[nodiscard]] MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &x) const override {
            double likelihood = std::transform_reduce(std::execution::seq,
                                                      models.begin(),
                                                      models.end(),
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
        computeLogLikelihoodGradient(const VectorType &x) const override {
            std::vector<double> weightedLikelihoods;

            std::transform(models.begin(),
                           models.end(),
                           weights.begin(),
                           std::back_inserter(weightedLikelihoods),
                           [&x](const std::shared_ptr<Model> &model, double weight) {
                               return weight * std::exp(-model->computeNegativeLogLikelihood(x));
                           }
            );

            double denominator = std::accumulate(weightedLikelihoods.begin(), weightedLikelihoods.end(), 0.);

            return std::transform_reduce(models.begin(),
                               models.end(),
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
        computeExpectedFisherInformation(const VectorType &x) const override {
            return std::nullopt;
        }

        std::unique_ptr<Model> deepCopy() const override {
            return std::make_unique<Mixture>(models, weights);
        }

    private:
        std::vector<std::shared_ptr<Model>> models;
        std::vector<double> weights;
    };
}

#endif //HOPS_MIXTURE_HPP
