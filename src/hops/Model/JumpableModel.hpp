#ifndef HOPS_JUMPABLEMODEL_HPP
#define HOPS_JUMPABLEMODEL_HPP

#include "Model.hpp"

namespace hops {

    template<typename ModelImpl>
    class JumpableModel : public Model {
    public:
        JumpableModel(const ModelImpl &modelImpl);

        double computeNegativeLogLikelihood(const VectorType &x) override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &type) override;

        bool hasConstantExpectedFisherInformation() override;

    private:
        ModelImpl modelImpl;
    };

    template<>
    class JumpableModel<std::unique_ptr<Model>> : public Model {
    public:
        JumpableModel(std::unique_ptr<Model> modelImpl) : modelImpl(std::move(modelImpl)) {}

        JumpableModel(const JumpableModel<std::unique_ptr<Model>> &other) {
            this->modelImpl = other.modelImpl->copyModel();
        }

        JumpableModel(JumpableModel<std::unique_ptr<Model>> &&other) {
            this->modelImpl = std::move(other.modelImpl);
        }

        JumpableModel<std::unique_ptr<Model>>& operator=(const JumpableModel<std::unique_ptr<Model>> &other) {
            this->modelImpl = other.modelImpl->copyModel();
            return *this;
        }

        JumpableModel<std::unique_ptr<Model>>& operator=(JumpableModel<std::unique_ptr<Model>> &&other) {
            this->modelImpl = other.modelImpl->copyModel();
            this->modelImpl = std::move(other.modelImpl);
            return *this;
        }

        double computeNegativeLogLikelihood(const VectorType &x) override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override;

        std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &x) override;

        std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &type) override;

        bool hasConstantExpectedFisherInformation() override;

    private:
        std::unique_ptr<Model> modelImpl;
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
