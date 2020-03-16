#ifndef HOPS_SBMLMODEL_HPP
#define HOPS_SBMLMODEL_HPP

#include <string>
#include <vector>

namespace hops {
    template<typename MatrixType, typename VectorType>
    class SbmlModel {
    public:

        MatrixType getConstraintMatrix() const {
            return constraintMatrix;
        }

        void setConstraintMatrix(MatrixType newConstraintMatrix) {
            SbmlModel::constraintMatrix = newConstraintMatrix;
        }

        VectorType getConstraintVector() const {
            return constraintVector;
        }

        void setConstraintVector(VectorType newConstraintVector) {
            SbmlModel::constraintVector = newConstraintVector;
        }

        VectorType getLowerBounds() const {
            return lowerBounds;
        }

        void setLowerBounds(VectorType newLowerBounds) {
            SbmlModel::lowerBounds = newLowerBounds;
        }

        VectorType getUpperBounds() const {
            return upperBounds;
        }

        void setUpperBounds(VectorType newUpperBounds) {
            SbmlModel::upperBounds = newUpperBounds;
        }

        [[nodiscard]] const std::vector<std::string> &getParameterNames() const {
            return parameterNames;
        }

        void setParameterNames(const std::vector<std::string> &newParameterNames) {
            SbmlModel::parameterNames = newParameterNames;
        }

        MatrixType getStoichiometry() const {
            return stoichiometry;
        }

        void setStoichiometry(MatrixType stoichiometry) {
            SbmlModel::stoichiometry = stoichiometry;
        }


    private:
        MatrixType stoichiometry;
        MatrixType constraintMatrix;
        VectorType constraintVector;
        VectorType lowerBounds;
        VectorType upperBounds;
        std::vector<std::string> parameterNames;
    };
}
#endif //HOPS_SBMLMODEL_HPP
