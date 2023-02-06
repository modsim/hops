#ifndef HOPS_JUMPABLEMODEL_HPP
#define HOPS_JUMPABLEMODEL_HPP

#include "Model.hpp"

namespace hops {

    template<typename ModelImpl>
    class JumpableModel : public Model {
    public:
        JumpableModel(const ModelImpl &modelImpl);

        double computeNegativeLogLikelihood(const VectorType &x) override;

        std::vector<std::string> getDimensionNames() const override;

        std::unique_ptr<Model> copyModel() const override;

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &type) override;

        bool hasConstantExpectedFisherInformation() override;

    private:
        ModelImpl modelImpl;
    };

    template<typename ModelImpl>
    double JumpableModel<ModelImpl>::computeNegativeLogLikelihood(const VectorType &x) {
        return modelImpl.computeNegativeLogLikelihood(x.bottomRows(x.rows() / 2));
    }

    template<typename ModelImpl>
    std::vector<std::string> JumpableModel<ModelImpl>::getDimensionNames() const {
        return modelImpl.getDimensionNames();
    }

    template<typename ModelImpl>
    std::unique_ptr<Model> JumpableModel<ModelImpl>::copyModel() const {
        return std::make_unique<JumpableModel<ModelImpl>>(*this);
    }

    template<typename ModelImpl>
    std::optional<VectorType> JumpableModel<ModelImpl>::computeLogLikelihoodGradient(const VectorType &x) {
        return modelImpl.computeLogLikelihoodGradient(x.bottomRows(x.rows() / 2));
    }

    template<typename ModelImpl>
    std::optional<MatrixType> JumpableModel<ModelImpl>::computeExpectedFisherInformation(const VectorType &x) {
        return modelImpl.computeExpectedFisherInformation(x.bottomRows(x.rows() / 2));
    }

    template<typename ModelImpl>
    bool JumpableModel<ModelImpl>::hasConstantExpectedFisherInformation() {
        return modelImpl.hasConstantExpectedFisherInformation();
    }

    template<typename ModelImpl>
    JumpableModel<ModelImpl>::JumpableModel(const ModelImpl &modelImpl):modelImpl(modelImpl) {}
}

#endif //HOPS_JUMPABLEMODEL_HPP
