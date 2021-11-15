#ifdef HOPS_GUROBI_FOUND

#include <Eigen/Core>

#include "LinearProgramGurobiImpl.hpp"
#include "GurobiEnvironmentSingleton.hpp"

namespace {
    std::vector<GRBVar> addVariablesToModel(GRBModel *model, size_t numberOfVariables) {
        std::vector<GRBVar> variables;
        for (size_t i = 0; i < numberOfVariables; ++i) {
            variables.emplace_back(model->addVar(
                    -GRB_INFINITY,
                    +GRB_INFINITY,
                    0,
                    GRB_CONTINUOUS,
                    "x_" + std::to_string(i))
            );
        }
        return variables;
    }

    void addLinearConstraints(const Eigen::MatrixXd &inequalityA,
                              const Eigen::VectorXd &inequalityB,
                              GRBModel *model,
                              const std::vector<GRBVar> &variables) {
        for (long i = 0; i < inequalityA.rows(); ++i) {
            GRBLinExpr expression;
            double coefficients[inequalityA.cols()];
            for (long j = 0; j < inequalityA.cols(); ++j) {
                coefficients[j] = inequalityA.coeff(i, j);
            }
            expression.addTerms(coefficients, &variables[0], inequalityA.cols());
            model->addConstr(expression, GRB_LESS_EQUAL, inequalityB(i), "row_" + std::to_string(i));
        }
    }

    void addObjective(const Eigen::VectorXd &objective, GRBModel *model,
                      const std::vector<GRBVar> &variables) {
        GRBLinExpr objectiveExpression = 0.0;
        objectiveExpression.addTerms(objective.data(), &variables[0], objective.rows());
        model->setObjective(objectiveExpression, GRB_MAXIMIZE);
    }

    hops::LinearProgramStatus parseGurobiStatus(int returnCode) {
        switch (returnCode) {
            case 2: {
                return hops::LinearProgramStatus::OPTIMAL;
            }
            case 3: {
                return hops::LinearProgramStatus::INFEASIBLE;
            }
            case 4: {
                return hops::LinearProgramStatus::UNDEFINED;
            }
            case 5: {
                return hops::LinearProgramStatus::UNBOUNDED;
            }
            default: {
                return hops::LinearProgramStatus::ERROR;
            }
        }
    }
}

hops::LinearProgramGurobiImpl::LinearProgramGurobiImpl(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) :
        LinearProgram(A, b),
        model(std::make_unique<GRBModel>(GRBModel(GurobiEnvironmentSingleton::getInstance().getGurobiEnvironment()))) {
    variables = addVariablesToModel(model.get(), A.cols());
    addLinearConstraints(A, b, model.get(), variables);
    model->update();
}

hops::LinearProgramGurobiImpl::LinearProgramGurobiImpl(const hops::LinearProgramGurobiImpl &other) :
        LinearProgram(other.A, other.b),
        variables(other.variables) {
    model = std::make_unique<GRBModel>(*other.model);
}

hops::LinearProgramGurobiImpl &hops::LinearProgramGurobiImpl::operator=(const hops::LinearProgramGurobiImpl &other) {
    this->A = other.A;
    this->b = other.b;
    this->model = std::make_unique<GRBModel>(*other.model);
    this->variables = other.variables;
    return *this;
}

hops::LinearProgramSolution hops::LinearProgramGurobiImpl::solve(const Eigen::VectorXd &objective) const {
    addObjective(objective, model.get(), variables);
    model->update();

    try {
        model->optimize();

        int status = model->get(GRB_IntAttr_Status);

        if (status == GRB_INF_OR_UNBD) {
            model->set(GRB_IntParam_Presolve, 0);
            model->optimize();
            status = model->get(GRB_IntAttr_Status);
        }
        if (status == GRB_OPTIMAL) {
            double objectiveValue = model->get(GRB_DoubleAttr_ObjVal);
            auto numberOfColumns = model->get(GRB_IntAttr_NumVars);
            auto modelVariables = model->getVars();
            Eigen::VectorXd solution(numberOfColumns);
            for (int i = 0; i < numberOfColumns; ++i) {
                solution(i) = modelVariables[i].get(GRB_DoubleAttr_X);
            }
            return LinearProgramSolution(objectiveValue, solution, parseGurobiStatus(status));
        } else if (status == GRB_INFEASIBLE) {
            return LinearProgramSolution(std::numeric_limits<double>::quiet_NaN(),
                                         Eigen::VectorXd(),
                                         LinearProgramStatus::INFEASIBLE);
        } else if (status == GRB_UNBOUNDED) {
            return LinearProgramSolution(std::numeric_limits<double>::quiet_NaN(),
                                         Eigen::VectorXd(),
                                         LinearProgramStatus::UNBOUNDED);
        } else if (status == GRB_UNDEFINED) {
            return LinearProgramSolution(std::numeric_limits<double>::quiet_NaN(),
                                         Eigen::VectorXd(),
                                         LinearProgramStatus::UNDEFINED);
        }
    }
    catch (const GRBException &e) {
        throw std::runtime_error("Gurobi encountered an exception: " + e.getMessage());
    }
    throw std::runtime_error("Exception: Gurobi failed to provide problem status or exception.");
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd>
hops::LinearProgramGurobiImpl::removeRedundantConstraints(double tolerance) {
    std::vector<long> constraintsToRemove;
    for (int i = 0; i < static_cast<int>(A.rows()); ++i) {
        GRBConstr constraintToTestForRedundancy = model->getConstrByName("row_" + std::to_string(i));
        model->remove(constraintToTestForRedundancy);
        model->update();
        GRBLinExpr constraintLHS;
        double coefficients[A.cols()];
        for (long j = 0; j < A.cols(); ++j) {
            coefficients[j] = A.coeff(i, j);
        }
        constraintLHS.addTerms(coefficients, &variables[0], A.cols());
        try {
            auto temporaryConstraint = model->addConstr(constraintLHS, GRB_LESS_EQUAL, b(i) + 10);
            model->update();
            auto solution = solve(A.row(i));
            model->remove(temporaryConstraint);
            model->update();

            if (solution.status != LinearProgramStatus::OPTIMAL || solution.objectiveValue + tolerance > b(i)) {
                model->addConstr(constraintLHS, GRB_LESS_EQUAL, b(i), "row_" + std::to_string(i));
                model->update();
            } else {
                constraintsToRemove.emplace_back(i);
            }
        }
        catch (GRBException &e) {
            std::cerr << "error code " << e.getErrorCode() << ": " << e.getMessage() << std::endl;
        }
    }

    std::vector<long> constraintsToKeep;
    for (long i = 0; i < A.rows(); ++i) {
        if (std::find(constraintsToRemove.begin(), constraintsToRemove.end(), i) == constraintsToRemove.end()) {
            constraintsToKeep.emplace_back(i);
        }
    }

    Eigen::MatrixXd newA(constraintsToKeep.size(), A.cols());
    Eigen::VectorXd newb(constraintsToKeep.size());
    for (size_t i = 0; i < constraintsToKeep.size(); ++i) {
        newb(i) = b(constraintsToKeep.at(i));
        newA.row(i) = A.row(constraintsToKeep.at(i));
    }

    *this = LinearProgramGurobiImpl(newA, newb);
    return std::make_tuple(A, b);
}

hops::LinearProgramSolution hops::LinearProgramGurobiImpl::computeChebyshevCenter() const {
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

    LinearProgramSolution chebyshevSolution = LinearProgramGurobiImpl(A_ext, b_ext).solve(obj);
    chebyshevSolution.optimalParameters.conservativeResize(A.cols());
    return chebyshevSolution;
}

std::vector<long> hops::LinearProgramGurobiImpl::computeUnconstrainedDimensions() const {
    std::vector<long> directions;
    for (long i = 0; i < A.cols(); ++i) {
        Eigen::VectorXd objective = Eigen::VectorXd::Zero(A.cols());
        objective(i) = 1.0;
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
hops::LinearProgramGurobiImpl::addBoxConstraintsToUnconstrainedDimensions(double lb, double ub) {
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
    *this = LinearProgramGurobiImpl(A, b);
    return std::make_tuple(A, b);
}

#endif //HOPS_GUROBI_FOUND
