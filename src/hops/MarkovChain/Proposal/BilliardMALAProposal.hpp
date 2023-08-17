#ifndef HOPS_BILLIARDMALA_HPP
#define HOPS_BILLIARDMALA_HPP

#include <Eigen/Eigenvalues>
#include <random>
#include <optional>
#include <utility>

#include "hops/Model/Model.hpp"
#include "hops/MarkovChain/ParallelTempering/Coldness.hpp"
#include "hops/Transformation/Transformation.hpp"
#include "hops/Utility/DefaultDimensionNames.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/LogSqrtDeterminant.hpp"
#include "hops/Utility/StringUtility.hpp"
#include "hops/Utility/VectorType.hpp"

#include "Proposal.hpp"
#include "Reflector.hpp"

namespace hops {

    /**
     * @tparam ModelType
     * @tparam InternalMatrixType
     */
    template<typename ModelType, typename InternalMatrixType>
    class BilliardMALAProposal : public Proposal, public ModelType {
    public:
        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        BilliardMALAProposal(InternalMatrixType A,
                             VectorType b,
                             const VectorType &currentState,
                             ModelType model,
                             long maxReflections,
                             double newStepSize = 1);


        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @param A
         * @param b
         * @param currentState
         */
        BilliardMALAProposal(const InternalMatrixType &A,
                             const VectorType &b,
                             const MatrixType &quadraticConstraintMatrix,
                             const VectorType &quadraticConstraintOffset,
                             double quadraticConstraintLhs,
                             const VectorType &currentState,
                             ModelType model,
                             long maxReflections,
                             double newStepSize = 1);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &propose(RandomNumberGenerator &rng, const Eigen::VectorXd &activeIndices) override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        void setProposal(const VectorType &newProposal) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override;

        [[nodiscard]] std::optional<double> getStepSize() const override;

        void setStepSize(double stepSize);

        [[nodiscard]] static bool hasStepSize();

        void setDimensionNames(const std::vector<std::string> &names) override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] double getStateNegativeLogLikelihood() override;

        [[nodiscard]] double getProposalNegativeLogLikelihood() override;

        [[nodiscard]] bool hasNegativeLogLikelihood() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        const MatrixType &getA() const override;

        const VectorType &getB() const override;

        [[nodiscard]] std::unique_ptr<Model> getModel() const;

    private:
        VectorType computeGradient(VectorType x);

        InternalMatrixType A;
        MatrixType Adense;
        VectorType b;
        std::optional<MatrixType> quadraticConstraintsMatrix;
        std::optional<VectorType> quadraticConstraintsOffset;
        std::optional<double> quadraticConstraintsLhs;

        VectorType state;
        VectorType driftedState;
        VectorType proposal;
        VectorType unreflectedProposal;
        VectorType driftedProposal;

        double stateLogSqrtDeterminant = 0;
        double proposalLogSqrtDeterminant = 0;
        double stateNegativeLogLikelihood = 0;
        double proposalNegativeLogLikelihood = 0;


        MatrixType stateMetric;
        MatrixType proposalMetric;
        Eigen::LLT<MatrixType> stateSolver;
        Eigen::LLT<MatrixType> proposalSolver;

        double stepSize = 1;
        double geometricFactor = 0;
        double covarianceFactor = 0;

        std::vector<std::string> dimensionNames;

        std::normal_distribution<double> normalDistribution{0., 1.};

        long maxNumberOfReflections;
    };

    template<typename ModelType, typename InternalMatrixType>
    BilliardMALAProposal<ModelType, InternalMatrixType>::BilliardMALAProposal(InternalMatrixType A,
                                                                              VectorType b,
                                                                              const VectorType &currentState,
                                                                              ModelType model,
                                                                              long maxReflections,
                                                                              double newStepSize) :
            ModelType(std::move(model)),
            A(std::move(A)),
            Adense(MatrixType(this->A)),
            b(std::move(b)),
            maxNumberOfReflections(maxReflections) {
        if (ModelType::hasConstantExpectedFisherInformation()) {
            computeGradient(currentState);
            stateMetric = ModelType::computeExpectedFisherInformation(currentState).value();
            stateSolver = Eigen::LLT<MatrixType>(stateMetric);
            if (stateSolver.info() != Eigen::Success) {
                throw std::runtime_error("cholesky decomposition of fisher information failed,"
                                         "because fisher information is not positive definite.");
            };
            stateLogSqrtDeterminant = logSqrtDeterminant(Eigen::MatrixXd(stateSolver.matrixL()));
        }
        BilliardMALAProposal::setState(currentState);
        BilliardMALAProposal::setStepSize(newStepSize);

        proposal = state;
        unreflectedProposal = state;
        proposalNegativeLogLikelihood = stateNegativeLogLikelihood;
        proposalMetric = stateMetric;
        proposalSolver = stateSolver;
        proposalLogSqrtDeterminant = stateLogSqrtDeterminant;

        if (!ModelType::getDimensionNames().empty()) {
            assert(ModelType::getDimensionNames().size() == this->state.rows());
            this->dimensionNames = ModelType::getDimensionNames();
        } else {
            this->dimensionNames = hops::createDefaultDimensionNames(this->state.rows());
        }
    }


    template<typename ModelType, typename InternalMatrixType>
    BilliardMALAProposal<ModelType, InternalMatrixType>::BilliardMALAProposal(const InternalMatrixType &A,
                                                                              const VectorType &b,
                                                                              const MatrixType &quadraticConstraintMatrix,
                                                                              const VectorType &quadraticConstraintOffset,
                                                                              double quadraticConstraintLhs,
                                                                              const VectorType &currentState,
                                                                              ModelType model,
                                                                              long maxReflections,
                                                                              double newStepSize) :
            BilliardMALAProposal(
                    A,
                    b,
                    currentState,
                    std::move(model),
                    maxReflections,
                    newStepSize) {
        this->quadraticConstraintsMatrix = quadraticConstraintMatrix;
        this->quadraticConstraintsOffset = quadraticConstraintOffset;
        this->quadraticConstraintsLhs = quadraticConstraintLhs;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &BilliardMALAProposal<ModelType, InternalMatrixType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            proposal(i) = normalDistribution(rng);
        }
        unreflectedProposal = driftedState + covarianceFactor * stateSolver.matrixU().solve(proposal);

        std::tuple<bool, long, VectorType> reflectionResult;
        if (quadraticConstraintsMatrix) {
            reflectionResult = Reflector::reflectIntoPolytope(Adense,
                                                              b,
                                                              quadraticConstraintsMatrix.value(),
                                                              quadraticConstraintsOffset.value(),
                                                              quadraticConstraintsLhs.value(),
                                                              state,
                                                              unreflectedProposal,
                                                              maxNumberOfReflections);
        } else {
            reflectionResult = Reflector::reflectIntoPolytope(Adense,
                                                              b,
                                                              state,
                                                              unreflectedProposal,
                                                              maxNumberOfReflections);
        }

        proposal = std::get<2>(reflectionResult);
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &BilliardMALAProposal<ModelType, InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        driftedState.swap(driftedProposal);
        stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
        unreflectedProposal = state;
        if (!ModelType::hasConstantExpectedFisherInformation()) {
            stateSolver = proposalSolver;
            stateMetric.swap(proposalMetric);
            stateLogSqrtDeterminant = proposalLogSqrtDeterminant;
        }
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setState(const VectorType &newState) {
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }

        state = newState;
        // Important: compute gradient before fisher info or else 13CFLUX2 will throw, since it uses internal
        // gradient data to construct fisher information.
        VectorType gradient = computeGradient(state);

        if (!ModelType::hasConstantExpectedFisherInformation()) {
            std::optional<decltype(stateMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                    state);
            if (optionalFisherInformation) {
                stateMetric = optionalFisherInformation.value();
            } else {
                stateMetric = MatrixType::Identity(state.rows(), state.rows());
            }

            stateSolver = Eigen::LLT<MatrixType>(stateMetric);
            if (stateSolver.info() != Eigen::Success) {
                throw std::runtime_error("cholesky decomposition of fisher information failed,"
                                         "because fisher information is not positive definite.");
            };
            stateLogSqrtDeterminant = logSqrtDeterminant(Eigen::MatrixXd(stateSolver.matrixL()));
        }

        driftedState = state + 0.5 * std::pow(covarianceFactor, 2) *
                               stateSolver.matrixU().template solve(stateSolver.matrixL().solve(gradient));

        stateNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setProposal(const VectorType &newProposal) {
        proposal = newProposal;
        proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposal);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::optional<double> BilliardMALAProposal<ModelType, InternalMatrixType>::getStepSize() const {
        return stepSize;
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
        geometricFactor = A.cols() / (2 * stepSize * stepSize);
        covarianceFactor = stepSize / std::sqrt(A.cols());
        setState(state);
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string BilliardMALAProposal<ModelType, InternalMatrixType>::getProposalName() const {
        return "BilliardMALA";
    }

    template<typename ModelType, typename InternalMatrixType>
    double BilliardMALAProposal<ModelType, InternalMatrixType>::getStateNegativeLogLikelihood() {
        return stateNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    double BilliardMALAProposal<ModelType, InternalMatrixType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (!isProposalInteriorPoint) {
            return -std::numeric_limits<double>::infinity();
        }
        // Important: compute gradient before fisher info or else x3cflux2 will throw
        VectorType gradient = computeGradient(proposal);

        if (!ModelType::hasConstantExpectedFisherInformation()) {
            std::optional<decltype(proposalMetric)> optionalFisherInformation = ModelType::computeExpectedFisherInformation(
                    proposal);
            if (optionalFisherInformation) {
                proposalMetric = optionalFisherInformation.value();
            } else {
                proposalMetric = MatrixType::Identity(state.rows(), state.rows());
            }

            proposalSolver = Eigen::LLT<MatrixType>(proposalMetric);
            if (proposalSolver.info() != Eigen::Success) {
                // state is not valid, because metric is not positive definite.
                return -std::numeric_limits<double>::infinity();
            };
            proposalLogSqrtDeterminant = logSqrtDeterminant(Eigen::MatrixXd(proposalSolver.matrixL()));
        }

        driftedProposal = proposal +
                          0.5 * std::pow(covarianceFactor, 2) *
                          proposalSolver.matrixU().template solve(
                                  proposalSolver.matrixL().solve(gradient));

        proposalNegativeLogLikelihood = ModelType::computeNegativeLogLikelihood(proposal);

        double normDifference =
                static_cast<double>((driftedState - unreflectedProposal).transpose() * stateMetric *
                                    (driftedState - unreflectedProposal)) -
                static_cast<double>((state - driftedProposal).transpose() * proposalMetric * (state - driftedProposal));

        return -proposalNegativeLogLikelihood
               + stateNegativeLogLikelihood
               + proposalLogSqrtDeterminant
               - stateLogSqrtDeterminant
               + geometricFactor * normDifference;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BilliardMALAProposal<ModelType, InternalMatrixType>::computeGradient(VectorType x) {
        std::optional<Eigen::VectorXd> gradient = ModelType::computeLogLikelihoodGradient(x);
        if (gradient) {
            return gradient.value();
        }
        return VectorType::Zero(x.rows());
    }

    template<typename ModelType, typename InternalMatrixType>
    bool BilliardMALAProposal<ModelType, InternalMatrixType>::hasStepSize() {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Proposal> BilliardMALAProposal<ModelType, InternalMatrixType>::copyProposal() const {
        return std::make_unique<BilliardMALAProposal>(*this);
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BilliardMALAProposal<ModelType, InternalMatrixType>::getState() const {
        return state;
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType BilliardMALAProposal<ModelType, InternalMatrixType>::getProposal() const {
        return proposal;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::vector<std::string> BilliardMALAProposal<ModelType, InternalMatrixType>::getParameterNames() const {
        return {"step_size", "max_reflections"};
    }

    template<typename ModelType, typename InternalMatrixType>
    std::any
    BilliardMALAProposal<ModelType, InternalMatrixType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(this->stepSize);
        }
        if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return std::any(this->maxNumberOfReflections);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename ModelType, typename InternalMatrixType>
    std::string
    BilliardMALAProposal<ModelType, InternalMatrixType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return "long";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setParameter(const ProposalParameter &parameter,
                                                                           const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            maxNumberOfReflections = std::any_cast<long>(value);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename ModelType, typename InternalMatrixType>
    double BilliardMALAProposal<ModelType, InternalMatrixType>::getProposalNegativeLogLikelihood() {
        return proposalNegativeLogLikelihood;
    }

    template<typename ModelType, typename InternalMatrixType>
    bool BilliardMALAProposal<ModelType, InternalMatrixType>::hasNegativeLogLikelihood() const {
        return true;
    }

    template<typename ModelType, typename InternalMatrixType>
    const MatrixType &BilliardMALAProposal<ModelType, InternalMatrixType>::getA() const {
        return Adense;
    }

    template<typename ModelType, typename InternalMatrixType>
    const VectorType &BilliardMALAProposal<ModelType, InternalMatrixType>::getB() const {
        return b;
    }

    template<typename ModelType, typename InternalMatrixType>
    std::unique_ptr<Model> BilliardMALAProposal<ModelType, InternalMatrixType>::getModel() const {
        return ModelType::copyModel();
    }

    template<typename ModelType, typename InternalMatrixType>
    VectorType &BilliardMALAProposal<ModelType, InternalMatrixType>::propose(RandomNumberGenerator &,
                                                                             const Eigen::VectorXd &) {
        throw std::runtime_error("Propose with rng and activeIndices not implemented");
    }

    template<typename ModelType, typename InternalMatrixType>
    void BilliardMALAProposal<ModelType, InternalMatrixType>::setDimensionNames(const std::vector<std::string> &names) {
        dimensionNames = names;
    }


    template<typename ModelType, typename InternalMatrixType>
    std::vector<std::string>
    BilliardMALAProposal<ModelType, InternalMatrixType>::getDimensionNames() const {
        return dimensionNames;
    }
}

#endif //HOPS_BILLIARDMALA_HPP
