#ifndef HOPS_DYNMULTIMODALMODEL_HPP
#define HOPS_DYNMULTIMODALMODEL_HPP

#include "MultivariateGaussianModel.hpp"
#include <vector>
#include <iostream>

namespace hops {
    template<typename Model>
    /**
     * @brief Dynamic multimodal model, which forms weighted linear combinations of arbirtrary models of the same type Model.
     * @details A dynamic multimodal model, which is a weighted linear combination of a set of models of the type Model, i.e.
     *          \f[ 
     *              g(x) = \sum_{i=0}^N w_i f_i(x)
     *          \f]
     *          These models have to implement the same functions as this class does.
     * @tparam Model The model type of the models which form the multimodal model.
     */
    class DynMultimodalModel {
    public:
        using MatrixType = typename Model::MatrixType;
        using VectorType = typename Model::VectorType;

        explicit DynMultimodalModel(
            const std::vector<Model>& modelComponents
        ) : modelComponents(modelComponents) {
            weights = std::vector<double>(modelComponents.size(), 1);
            assert(weights.size() == modelComponents.size());
        }

        explicit DynMultimodalModel(
            const std::vector<Model>& modelComponents, const std::vector<double> weights
       ) : modelComponents(modelComponents), weights(weights) 
        {
            assert(weights.size() == modelComponents.size());
        }

        /**
         * @brief Computes the negative log likelihood of the multimodal model.
         * @details Assume \f$ g = g(x) \f$ is the density of the multimodal model as defined above. 
         * This method computes the negative log likelihood as
         *          \f{align*}{
         *              -\log g &= -\log \sum_{i=0}^N w_i f_i \\
         *              &= -\log \sum_{i=0}^N w_i \exp\{ \log f_i \} \\
         *          \f}
         *          using the components `calculateNegativeLogLikelihood` method, which has to be provided
         *          by `Model`.
         * @param x 
         * @return
         */
        typename MatrixType::Scalar calculateNegativeLogLikelihood(const VectorType &x) const {
            typename MatrixType::Scalar y = 0;
            for (size_t i = 0; i < modelComponents.size(); ++i) {
                y += weights[i] * std::exp(-modelComponents[i].calculateNegativeLogLikelihood(x));
            }
            return -std::log(y);
        }
        
        /**
         * @brief Not available, throws exception on call.
         * @details There seems to exist no easy to compute closed form expression for the expected fisher information of a linear combination
         *          of densities, where the expected fisher information is known for each density individually. 
         *          This function is thus not yet implemented and throws an exception.
         */
        MatrixType calculateExpectedFisherInformation(const VectorType &) const {
            throw std::string("No closed form expression available and thus not yet implemented.");
        }

        /**
         * @brief Computes the negative log likelihood of the multimodal model.
         * @details Assume \f$ g = g(x) \f$ is the density of the multimodal model as defined above. 
         * This method computes the gradient of the log-likelihood as
         *          \f{align*}{
         *               \nabla \log g 
         *               &= \frac{ \nabla g }{ g }
         *               &= \frac{ \nabla \sum_{i=0}^N w_i f_i }{ \sum_{i=0}^N w_i f_i }
         *               &= \frac{ \sum_{i=0}^N w_i \nabla \exp\{\log f_i\} }{ \sum_{i=0}^N w_i \exp\{\log f_i\} }
         *               &= \frac{ \sum_{i=0}^N w_i \exp\{\log f_i\} \nabla \log \fi }{ \sum_{i=0}^N w_i \exp\{\log f_i\} }
         *          \f}
         *          Note that \f$ -\log f_i \f$ and \f$\nabla \log f_i \f$ have to be provided by
         *          the components in `modelComponents` by the `calculateNegativeLogLikelihood` and `calculateLogLikelihoodGradient` methods.
         * @param x 
         * @return
         */
        VectorType calculateLogLikelihoodGradient(const VectorType &x) const {
            VectorType 
                gradY = weights[0] * modelComponents[0].calculateLogLikelihoodGradient(x) * std::exp(-modelComponents[0].calculateNegativeLogLikelihood(x));
            typename MatrixType::Scalar denominator = weights[0] * std::exp(-modelComponents[0].calculateNegativeLogLikelihood(x));
            for (size_t i = 1; i < modelComponents.size(); ++i) {
                gradY += weights[i] * modelComponents[i].calculateLogLikelihoodGradient(x) * std::exp(-modelComponents[i].calculateNegativeLogLikelihood(x)),
                denominator += weights[i] * std::exp(-modelComponents[i].calculateNegativeLogLikelihood(x));
            }
            return gradY / denominator;
        }

    private:
        std::vector<Model> modelComponents;
        std::vector<double> weights;    
    };
}

#endif //HOPS_DYNMULTIMODALMODEL_HPP
