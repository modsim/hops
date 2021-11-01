#ifndef HOPS_MODELWRAPPER_HPP
#define HOPS_MODELWRAPPER_HPP

#include <hops/Model/Model.hpp>

namespace hops {

    /**
     * @Brief Wrapper to help mix in models to proposals. It also allows for swapping models dynamically.
     */
    class ModelWrapper {
    public:
        explicit ModelWrapper(std::shared_ptr<Model> model) : model(std::move(model)) {}

        /**
         * @Brief Virtual because later mixins are allowed to override, e.g. Coldness
         */
        [[nodiscard]] virtual MatrixType::Scalar computeNegativeLogLikelihood(const VectorType &state)  {
            return model->computeNegativeLogLikelihood(state);
        }

        /**
         * @Brief Virtual because later mixins are allowed to override, e.g. Coldness
         */
        [[nodiscard]] virtual std::optional<VectorType> computeLogLikelihoodGradient(const VectorType &state)  {
                return model->computeLogLikelihoodGradient(state);
        }

        /**
         * @Brief Virtual because later mixins are allowed to override, e.g. Coldness
         */
        [[nodiscard]] virtual std::optional<MatrixType> computeExpectedFisherInformation(const VectorType &state)  {
                return model->computeExpectedFisherInformation(state);
        }

        [[nodiscard]] const std::shared_ptr<Model> &getModel() const {
            return model;
        }

        void setModel(const std::shared_ptr<Model> &newModel) {
            ModelWrapper::model = newModel;
        }

        bool hasModel() {
            return model != nullptr;
        }

    private:
        std::shared_ptr<Model> model;
    };
}

#endif //HOPS_MODELWRAPPER_HPP
