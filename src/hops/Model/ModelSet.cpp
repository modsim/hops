#include "ModelSet.hpp"

double hops::ModelSet::computeNegativeLogLikelihood(const hops::VectorType &x) {
    return 0;
}

std::vector<std::string> hops::ModelSet::getDimensionNames() const {
    return std::vector<std::string>();
}

std::unique_ptr<Model> hops::ModelSet::copyModel() const {
    return std::unique_ptr<Model>();
}
