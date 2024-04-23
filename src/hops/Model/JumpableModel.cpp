#include "JumpableModel.hpp"

std::optional<hops::MatrixType> hops::JumpableModel<std::unique_ptr<hops::Model>>::computeExpectedFisherInformation(const VectorType &x) {
    return modelImpl->computeExpectedFisherInformation(x.bottomRows(x.rows() / 2));
}

std::unique_ptr<hops::Model> hops::JumpableModel<std::unique_ptr<hops::Model>>::copyModel() const {
    return std::make_unique<JumpableModel<std::unique_ptr<Model>>>(modelImpl->copyModel());
}

std::optional<hops::VectorType> hops::JumpableModel<std::unique_ptr<hops::Model>>::computeLogLikelihoodGradient(const VectorType &x) {
    return modelImpl->computeLogLikelihoodGradient(x.bottomRows(x.rows() / 2));
}

bool hops::JumpableModel<std::unique_ptr<hops::Model>>::hasConstantExpectedFisherInformation() {
    return modelImpl->hasConstantExpectedFisherInformation();
}

double hops::JumpableModel<std::unique_ptr<hops::Model>>::computeNegativeLogLikelihood(const VectorType &x) {
    return modelImpl->computeNegativeLogLikelihood(x.bottomRows(x.rows() / 2));
}

std::vector<std::string> hops::JumpableModel<std::unique_ptr<hops::Model>>::getDimensionNames() const {
    return modelImpl->getDimensionNames();
}
