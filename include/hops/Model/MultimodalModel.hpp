#ifndef HOPS_MULTIMODALMODEL_HPP
#define HOPS_MULTIMODALMODEL_HPP

#include "MultivariateGaussianModel.hpp"
#include <vector>

namespace hops {
    template<typename ...Models>
    /**
     * @details Uses MatrixType and VectorType types as defined by the first Model type.
     * @tparam Models
     */
    class MultimodalModel {
    public:
        using MatrixType = typename std::tuple_element<0, std::tuple<Models...>>::type::MatrixType;
        using VectorType = typename std::tuple_element<0, std::tuple<Models...>>::type::VectorType;

        explicit MultimodalModel(const std::tuple<Models...> &modelComponents) : modelComponents(modelComponents) {

        }

        typename MatrixType::Scalar calculateNegativeLogLikelihood(const VectorType &x) const {
            return std::apply(
                    [&x](auto &... args) {
                        return -std::log(((1. / std::tuple_size<decltype(modelComponents)>::value *
                                           std::exp(-args.calculateNegativeLogLikelihood(x))) + ...));
                    },
                    modelComponents);
        }

        MatrixType calculateExpectedFisherInformation(const VectorType &x) const {
            return std::apply(
                    [&x](auto &... args) {
                        // MatrixType required to prevent std::bad_alloc in debug mode
                        return MatrixType(((args.calculateExpectedFisherInformation(x)) + ...));
                    },
                    modelComponents);
        }

        VectorType calculateLogLikelihoodGradient(const VectorType &x) const {
            return std::apply(
                    [&x](auto &... args) {
                        // VectorType required to prevent std::bad_alloc in debug mode
                        return VectorType(((args.calculateLogLikelihoodGradient(x)) + ...));
                    },
                    modelComponents);
        }

    private:
        std::tuple<Models...> modelComponents;
    };
}

#endif //HOPS_MULTIMODALMODEL_HPP
