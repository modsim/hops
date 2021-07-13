#ifndef HOPS_UNIFORMDUMMYMODEL_HPP
#define HOPS_UNIFORMDUMMYMODEL_HPP

#include <stdexcept>
namespace hops {
    /**
     * @details Uses MatrixType and VectorType types as defined by the first Model type.
     * @tparam Models
     */
    template<typename Matrix, typename Vector>
    class UniformDummyModel {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;

        UniformDummyModel() {}

        typename MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &) const {
            throw std::runtime_error("UniformDummyModel is not supposed to be actually used.");
        }

        MatrixType computeExpectedFisherInformation(const VectorType &) const {
            throw std::runtime_error("UniformDummyModel is not supposed to be actually used.");
        }

        VectorType computeLogLikelihoodGradient(const VectorType &) const {
            throw std::runtime_error("UniformDummyModel is not supposed to be actually used.");
        }
    };
}

#endif //HOPS_UNIFORMDUMMYMODEL_HPP
