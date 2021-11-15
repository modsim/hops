#ifdef HOPS_CLP_FOUND

#include <coin/ClpPackedMatrix.hpp>
#include <coin/ClpSimplex.hpp>

#include <Eigen/Sparse>

#include "LinearProgramClpImpl.hpp"
#include "LinearProgramStatus.hpp"

namespace {
    hops::LinearProgramStatus parseClpStatus(int returnCode) {
        switch (returnCode) {
            case 0:
                return hops::LinearProgramStatus::OPTIMAL;
            case 1:
                return hops::LinearProgramStatus::INFEASIBLE;
            case 2:
                return hops::LinearProgramStatus::UNBOUNDED;
            default:
                return hops::LinearProgramStatus::ERROR;
        }
    }

    bool isInequalityRedundant(const Eigen::MatrixXd &A,
                               const Eigen::VectorXd &b,
                               unsigned int index,
                               double tolerance) {
        Eigen::VectorXd bTemp = b;
        bTemp(index) += 1.0;

        auto result = hops::LinearProgramClpImpl(A, bTemp).solve(A.row(index));

        if (result.status != hops::LinearProgramStatus::OPTIMAL) {
            return false;
        }

        return result.objectiveValue <= b(index) + tolerance;
    }

}

hops::LinearProgramClpImpl::LinearProgramClpImpl(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) :
        LinearProgram(A, b) {
    if (A.cols() > std::numeric_limits<int>::max() || A.cols() < std::numeric_limits<int>::min()) {
        throw std::runtime_error(
                "Objective has more columns, than can fit into an integer, making it incompatible to CLP");
    }
    if (A.rows() > std::numeric_limits<int>::max() || A.rows() < std::numeric_limits<int>::min()) {
        throw std::runtime_error(
                "Objective has more rows, than can fit into an integer, making it incompatible to CLP");
    }

    std::vector<int> rowIndices;
    std::vector<int> columnIndices;
    std::vector<double> values;
    Eigen::SparseMatrix<double> sparseA = A.sparseView();
    for (long i = 0; i < A.outerSize(); ++i) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(sparseA, i); it; ++it) {
            if (it.value() != 0) {
                rowIndices.emplace_back(it.row());
                columnIndices.emplace_back(it.col());
                values.emplace_back(it.value());
            }
        }
    }

    CoinPackedMatrix matrix(false, rowIndices.data(), columnIndices.data(), values.data(), values.size());

    Eigen::VectorXd rowLower = Eigen::VectorXd::Constant(b.rows(), -DBL_MAX);
    Eigen::VectorXd colLower = Eigen::VectorXd::Constant(A.cols(), -DBL_MAX);
    Eigen::VectorXd colUpper = Eigen::VectorXd::Constant(A.cols(), DBL_MAX);

    model.loadProblem(matrix, colLower.data(), colUpper.data(), nullptr,
                      rowLower.data(), b.data());
    model.setOptimizationDirection(-1.);
    model.scaling(4);
    model.setLogLevel(0);
}

hops::LinearProgramClpImpl::LinearProgramClpImpl(const hops::LinearProgramClpImpl &other) :
        LinearProgram(other.A, other.b),
        model(other.model) {}

hops::LinearProgramClpImpl &hops::LinearProgramClpImpl::operator=(const hops::LinearProgramClpImpl &other) {
    this->A = other.A;
    this->b = other.b;
    this->model = other.model;
    return *this;
}

hops::LinearProgramSolution hops::LinearProgramClpImpl::solve(const Eigen::VectorXd &objective) const {
    for (int i = 0; i < static_cast<int>(objective.rows()); ++i) {
        model.setObjectiveCoefficient(i, objective(i));
    }
    model.primal();
    model.checkSolution();
    model.checkUnscaledSolution();

    return LinearProgramSolution(-model.rawObjectiveValue(), // - due to optimization direction internally
                                 Eigen::Map<Eigen::VectorXd>(model.primalColumnSolution(), objective.rows()),
                                 parseClpStatus(model.status()));
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> hops::LinearProgramClpImpl::removeRedundantConstraints(double tolerance) {
    if (A.rows() <= 1) {
        return std::make_tuple(A, b);
    }

    for (int i = 0; i < A.rows(); ++i) {
        int numRows = A.rows();
        // Try to remove ith inequality
        if (isInequalityRedundant(A, b, i, tolerance)) {
            if (i != numRows - 1) {
                // Swap the row which is going to be removed with the
                // last row
                A.row(i).swap(A.row(numRows - 1));
                double temp = b(i);
                b(i) = b(numRows - 1);
                b(numRows - 1) = temp;
                i--; // resets index after swap
            }

            // Remove the last row
            A.conservativeResize(numRows - 1, Eigen::NoChange);
            b.conservativeResize(numRows - 1);
        }
    }
    *this = LinearProgramClpImpl(A, b);
    return std::make_tuple(A, b);
}

hops::LinearProgramSolution hops::LinearProgramClpImpl::computeChebyshevCenter() const {
    //Extend system by dimension for radius
    const long numberOfRows = A.rows();
    const long numberOfColumns = A.cols();
    Eigen::MatrixXd A_ext = A;
    Eigen::MatrixXd l_col(numberOfRows + 1, 1);
    l_col << A.rowwise().norm(), -1;
    A_ext.conservativeResize(numberOfRows + 1, numberOfColumns + 1);
    A_ext.row(numberOfRows) = Eigen::VectorXd::Zero(numberOfColumns + 1);
    A_ext.col(numberOfColumns) = l_col;
    Eigen::VectorXd b_ext = b;
    b_ext.conservativeResize(numberOfRows + 1);
    b_ext(numberOfRows) = 0;

    //make objective
    Eigen::VectorXd obj = Eigen::VectorXd::Zero(numberOfColumns + 1);
    obj(numberOfColumns) = 1;

    LinearProgramSolution chebyshevSolution = LinearProgramClpImpl(A_ext, b_ext).solve(obj);
    chebyshevSolution.optimalParameters.conservativeResize(A.cols());
    return chebyshevSolution;
}

std::vector<long> hops::LinearProgramClpImpl::computeUnconstrainedDimensions() const {
    std::vector<long> directions;
    for (long i = 0; i < A.cols(); ++i) {
        Eigen::VectorXd objective = Eigen::VectorXd::Zero(A.cols());
        objective(i) = 1.0;
        for (int j = 0; j < static_cast<int>(objective.rows()); ++j) {
            model.setObjectiveCoefficient(j, objective(j));
        }
        auto forwardSolution = solve(objective);
        if (forwardSolution.status != hops::LinearProgramStatus::OPTIMAL) {
            directions.push_back(i + 1);
        }

        auto backwardSolution = solve(-objective);
        if (backwardSolution.status != hops::LinearProgramStatus::OPTIMAL) {
            directions.push_back(-i - 1);
        }
    }
    return directions;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
hops::LinearProgramClpImpl::addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) {
    std::vector<long> unconstrainedDimensions = computeUnconstrainedDimensions();

    for (const auto &unconstrainedDimension : unconstrainedDimensions) {
        A.conservativeResize(A.rows() + 1, A.cols());
        A.row(A.rows() - 1) = Eigen::VectorXd::Zero(A.cols());
        b.conservativeResize(b.rows() + 1);
        if (unconstrainedDimension > 0) {
            A(A.rows() - 1, unconstrainedDimension - 1) = 1;
            b(b.rows() - 1) = ub;
        } else {
            A(A.rows() - 1, -unconstrainedDimension - 1) = -1;
            b(b.rows() - 1) = lb;
        }
    }
    *this = LinearProgramClpImpl(A, b);
    return std::make_tuple(A, b);
}

#endif //HOPS_CLP_FOUND
