#ifndef HOPS_BILLIARDWALKPROPOSAL_HPP
#define HOPS_BILLIARDWALKPROPOSAL_HPP

#include <optional>
#include <random>
#include <utility>

#include "hops/Utility/DefaultDimensionNames.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/StringUtility.hpp"
#include "hops/Utility/VectorType.hpp"

#include "ChordStepDistributions.hpp"
#include "Proposal.hpp"
#include "Reflector.hpp"

namespace hops {

    /**
     * @tparam ModelType
     * @tparam InternalMatrixType
     */
    template<typename InternalMatrixType>
    class BilliardWalkProposal : public Proposal {
    public:
        /**
         * @brief Constructs proposal mechanism on polytope defined as Ax<b.
         * @details See 10.4230/LIPIcs.SoCG.2021.21 for more informatio on this algorithm
         * @param A
         * @param b
         * @param currentState
         */
        BilliardWalkProposal(InternalMatrixType A,
                             VectorType b,
                             const VectorType &currentState,
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

        bool isReflectionSuccessful() const;

        long getNumberOfReflections() const;

        void setStepSize(double stepSize);

        [[nodiscard]] static bool hasStepSize();

        void setDimensionNames(const std::vector<std::string> &names) override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const ProposalParameter &parameter) const override;

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override;

        void setParameter(const ProposalParameter &parameter, const std::any &value) override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

        [[nodiscard]] double computeLogAcceptanceProbability() override;

        const MatrixType &getA() const override;

        const VectorType &getB() const override;

    private:
        InternalMatrixType A;
        MatrixType Adense;
        VectorType b;

        VectorType state;
        VectorType proposal;
        VectorType unreflectedProposal;

        double stepSize = 1;        // called tau in the paper
        long maxNumberOfReflections;// called rho in the paper

        bool reflectionSuccessful;
        long numberOfReflections;

        VectorType updateDirection;
        double step = 0;
        UniformStepDistribution<VectorType::Scalar> chordStepDistribution;

        std::vector<std::string> dimensionNames;

        std::normal_distribution<double> normalDistribution{0., 1.};
    };

    /*
     * Recommendation for step size is roughly the diameter of the polytope
     */
    template<typename InternalMatrixType>
    BilliardWalkProposal<InternalMatrixType>::BilliardWalkProposal(InternalMatrixType A,
                                                                              VectorType b,
                                                                              const VectorType &currentState,
                                                                              long maxReflections,
                                                                              double newStepSize) : A(std::move(A)),
                                                                                                    Adense(MatrixType(this->A)),
                                                                                                    b(std::move(b)),
                                                                                                    maxNumberOfReflections(maxReflections) {
        BilliardWalkProposal::setState(currentState);
        BilliardWalkProposal::setStepSize(newStepSize);

        proposal = state;
        updateDirection = state;

        this->dimensionNames = hops::createDefaultDimensionNames(this->state.rows());
    }


    template<typename InternalMatrixType>
    VectorType &BilliardWalkProposal<InternalMatrixType>::propose(RandomNumberGenerator &rng) {
        for (long i = 0; i < proposal.rows(); ++i) {
            updateDirection(i) = normalDistribution(rng);
        }
        updateDirection.normalize();
        step = -this->stepSize * std::log(this->chordStepDistribution.draw(rng, 0, 1));
        proposal = state + step*updateDirection;


        std::tuple<bool, long, VectorType> reflectionResult = Reflector::reflectIntoPolytope(Adense,
                                                                                             b,
                                                                                             state,
                                                                                             proposal,
                                                                                             maxNumberOfReflections);

        reflectionSuccessful = std::get<0>(reflectionResult);
        numberOfReflections = std::get<1>(reflectionResult);
        proposal = std::get<2>(reflectionResult);
        return proposal;
    }

    template<typename InternalMatrixType>
    VectorType &BilliardWalkProposal<InternalMatrixType>::acceptProposal() {
        state.swap(proposal);
        return state;
    }

    template<typename InternalMatrixType>
    void BilliardWalkProposal<InternalMatrixType>::setState(const VectorType &newState) {
        if (((b - A * newState).array() < 0).any()) {
            throw std::invalid_argument("Starting point outside polytope always gives constant Markov chain.");
        }
        state = newState;
    }

    template<typename InternalMatrixType>
    void BilliardWalkProposal<InternalMatrixType>::setProposal(const VectorType &newProposal) {
        proposal = newProposal;
    }

    template<typename InternalMatrixType>
    std::optional<double> BilliardWalkProposal<InternalMatrixType>::getStepSize() const {
        return stepSize;
    }

    template<typename InternalMatrixType>
    void BilliardWalkProposal<InternalMatrixType>::setStepSize(double newStepSize) {
        stepSize = newStepSize;
    }

    template<typename InternalMatrixType>
    std::string BilliardWalkProposal<InternalMatrixType>::getProposalName() const {
        return "BilliardWalk";
    }

    template<typename InternalMatrixType>
    double BilliardWalkProposal<InternalMatrixType>::computeLogAcceptanceProbability() {
        bool isProposalInteriorPoint = ((A * proposal - b).array() < 0).all();
        if (not isProposalInteriorPoint || not this->reflectionSuccessful) {
            return -std::numeric_limits<double>::infinity();
        }
        return 0;
    }

    template<typename InternalMatrixType>
    bool BilliardWalkProposal<InternalMatrixType>::hasStepSize() {
        return true;
    }

    template<typename InternalMatrixType>
    std::unique_ptr<Proposal> BilliardWalkProposal<InternalMatrixType>::copyProposal() const {
        return std::make_unique<BilliardWalkProposal>(*this);
    }

    template<typename InternalMatrixType>
    VectorType BilliardWalkProposal<InternalMatrixType>::getState() const {
        return state;
    }

    template<typename InternalMatrixType>
    VectorType BilliardWalkProposal<InternalMatrixType>::getProposal() const {
        return proposal;
    }

    template<typename InternalMatrixType>
    std::vector<std::string> BilliardWalkProposal<InternalMatrixType>::getParameterNames() const {
        return {"step_size", "max_reflections"};
    }

    template<typename InternalMatrixType>
    std::any
    BilliardWalkProposal<InternalMatrixType>::getParameter(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return std::any(this->stepSize);
        }
        if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return std::any(this->maxNumberOfReflections);
        }
        throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
    }

    template<typename InternalMatrixType>
    std::string
    BilliardWalkProposal<InternalMatrixType>::getParameterType(const ProposalParameter &parameter) const {
        if (parameter == ProposalParameter::STEP_SIZE) {
            return "double";
        } else if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            return "long";
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType>
    void BilliardWalkProposal<InternalMatrixType>::setParameter(const ProposalParameter &parameter,
                                                                           const std::any &value) {
        if (parameter == ProposalParameter::STEP_SIZE) {
            setStepSize(std::any_cast<double>(value));
        } else if (parameter == ProposalParameter::MAX_REFLECTIONS) {
            maxNumberOfReflections = std::any_cast<long>(value);
        } else {
            throw std::invalid_argument("Can't get parameter which doesn't exist in " + this->getProposalName());
        }
    }

    template<typename InternalMatrixType>
    const MatrixType &BilliardWalkProposal<InternalMatrixType>::getA() const {
        return Adense;
    }

    template<typename InternalMatrixType>
    const VectorType &BilliardWalkProposal<InternalMatrixType>::getB() const {
        return b;
    }


    template<typename InternalMatrixType>
    VectorType &BilliardWalkProposal<InternalMatrixType>::propose(RandomNumberGenerator &,
                                                                             const Eigen::VectorXd &) {
        throw std::runtime_error("Propose with rng and activeIndices not implemented");
    }

    template<typename InternalMatrixType>
    void BilliardWalkProposal<InternalMatrixType>::setDimensionNames(const std::vector<std::string> &names) {
        dimensionNames = names;
    }

    template<typename InternalMatrixType>
    std::vector<std::string>
    BilliardWalkProposal<InternalMatrixType>::getDimensionNames() const {
        return dimensionNames;
    }
    template<typename InternalMatrixType>
    bool BilliardWalkProposal<InternalMatrixType>::isReflectionSuccessful() const {
        return reflectionSuccessful;
    }
    template<typename InternalMatrixType>
    long BilliardWalkProposal<InternalMatrixType>::getNumberOfReflections() const {
        return numberOfReflections;
    }
}// namespace hops

#endif//HOPS_BILLIARDWALKPROPOSAL_HPP
