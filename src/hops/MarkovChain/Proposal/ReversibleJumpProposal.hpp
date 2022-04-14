#ifndef HOPS_REVERSIBLEJUMPPROPOSAL_HPP
#define HOPS_REVERSIBLEJUMPPROPOSAL_HPP

#include <Eigen/Core>
#include <string>
#include <utility>
#include <vector>
#include <random>

#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>
#include <hops/Utility/StringUtility.hpp>

#include "Proposal.hpp"


// TODO list
// change default values from Eigen::VectorXd to collection?
// MarkovCHainImpl -> proposalImpl

namespace {
    std::pair<double, double>
    distanceInCoordinateDirection(const Eigen::MatrixXd &A,
                                  const Eigen::VectorXd &b,
                                  const Eigen::VectorXd &x,
                                  long coordinate) {
        Eigen::VectorXd slacks = b - A * x;
        Eigen::VectorXd inverseDistances = A.col(coordinate).cwiseQuotient(slacks);
        double forwardDistance = 1. / inverseDistances.maxCoeff();
        double backwardDistance = 1. / inverseDistances.minCoeff();

        for (long i = 0; i < inverseDistances.rows(); ++i) {
            if (inverseDistances(i) == -std::numeric_limits<double>::infinity()) {
                backwardDistance = 0;
            } else if (inverseDistances(i) == std::numeric_limits<double>::infinity()) {
                forwardDistance = 0;
            }
        }
        assert(backwardDistance <= 0 && forwardDistance >= 0);
        return std::make_pair(backwardDistance, forwardDistance);
    }
}

namespace hops {
    // TODO add proposal info to track whether it was model jump or parameter jump


    /**
     * @tparam ProposalImpl is required to have a model mixed in already.
     */
    template<typename ProposalImpl>
    class ReversibleJumpProposal : public ProposalImpl, public Proposal {
    public:
        ReversibleJumpProposal(const ProposalImpl &proposalImpl,
                               Eigen::VectorXi jumpIndices,
                               VectorType parameterDefaultValues);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &proposeModel(RandomNumberGenerator &rng) override;

        VectorType &proposeParameters(RandomNumberGenerator &rng) override;

        double computeLogAcceptanceProbability() override;

        VectorType &acceptProposal() override;

        void setState(const VectorType &state) override;

        [[nodiscard]] VectorType getState() const override;

        [[nodiscard]] VectorType getProposal() const override;

        [[nodiscard]] std::vector<std::string> getParameterNames() const override;

        [[nodiscard]] std::any getParameter(const std::string &parameterName) const override;

        [[nodiscard]] std::string getParameterType(const std::string &name) const override;

        void setParameter(const std::string &parameterName, const std::any &value) override;

        [[nodiscard]] bool hasStepSize() const override;

        [[nodiscard]] std::string getProposalName() const override;

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override;

    private:
        // fixed value from https://doi.org/10.1093/bioinformatics/btz500
        VectorType::Scalar modelJumpProbability = 0.5;
        VectorType::Scalar parameterActivationProbability = 0.1;
        VectorType::Scalar parameterDeactivationProbability = 0.1;
        VectorType::Scalar stepSize = 0.1;
        std::uniform_real_distribution<double> uniformRealDistribution;
        hops::GaussianStepDistribution<double> gaussianStepDistribution;

        Eigen::VectorXi jumpIndices_;

        Eigen::VectorXi jumpIndices;
        Eigen::VectorXd defaultValues;

        // activation also contains info about parameters that are not jumped.
        // This should make it easier to work with proposals down the line
        Eigen::VectorXd activationProposal;
        Eigen::VectorXd activationState;

        Eigen::VectorXd proposal;
        Eigen::VectorXd state;

        double logProposalAcceptanceProbability;

        /**
         *
         * @param randomNumberGenerator
         * @param proposalActivationState
         * @param state
         * @param defaultValues
         * @return modelJumpProb, activationProposal, proposal
         */
        std::tuple<double, VectorType, VectorType> jumpModel(RandomNumberGenerator &randomNumberGenerator,
                                                             const VectorType &activationState,
                                                             const VectorType &state,
                                                             const VectorType &defaultValues);
    };

    template<typename ProposalImpl>
    ReversibleJumpProposal<ProposalImpl>::ReversibleJumpProposal(const ProposalImpl &proposalImpl,
                                                                 Eigen::VectorXi jumpIndices,
                                                                 VectorType parameterDefaultValues) :
            ProposalImpl(proposalImpl),
            jumpIndices(std::move(jumpIndices)),
            defaultValues(std::move(parameterDefaultValues)) {

        activationState = Eigen::VectorXd::Ones(defaultValues.rows());

        // Occam's Razor: Starts with all optional parameters deactivated
        VectorType parameterState = ProposalImpl::getState();
        for (long i = 0; i < jumpIndices.rows(); ++i) {
            activationState(jumpIndices(i)) = 0;
            parameterState[jumpIndices(i)] = defaultValues[jumpIndices(i)];
        }
        activationProposal = activationState;
        ProposalImpl::setState(parameterState);
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposal<ProposalImpl>::propose(RandomNumberGenerator &rng) {
        if (uniformRealDistribution(rng) < modelJumpProbability) {
            return proposeModel(rng);
        } else {
            return proposeParameters(rng);
        }
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposal<ProposalImpl>::proposeParameters(RandomNumberGenerator &rng) {
        ProposalImpl::propose(rng, activationState);
        // TODO return
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposal<ProposalImpl>::proposeModel(RandomNumberGenerator &rng) {
        VectorType proposal = ProposalImpl::getState();

        // TODO return activationProposal & logModelJumpProbability
        double logModelJumpProbabilityDifferential = jumpModel(rng,
                                                               activationState,
                                                               state,
                                                               defaultValues);

        logProposalAcceptanceProbability = ProposalImpl::getStateNegativeLogLikelihood() -
                                           ProposalImpl::getProposalNegativeLogLikelihood() +
                                           logModelJumpProbabilityDifferential;


    }

    template<typename ProposalImpl>
    double ReversibleJumpProposal<ProposalImpl>::computeLogAcceptanceProbability() {
        return logProposalAcceptanceProbability;
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposal<ProposalImpl>::acceptProposal() {
        state.swap(proposal);
        activationState.swap(activationProposal);
        return state;
    }

    template<typename ProposalImpl>
    void ReversibleJumpProposal<ProposalImpl>::setState(const VectorType &newState) {
        if (state.rows() != newState.rows()) {
            throw std::invalid_argument("Setting state failed because new state does not have correct dimensions.");
        }
        state = newState;
    }

    template<typename ProposalImpl>
    VectorType ReversibleJumpProposal<ProposalImpl>::getState() const {
        return state;
    }

    template<typename ProposalImpl>
    VectorType ReversibleJumpProposal<ProposalImpl>::getProposal() const {
        return proposal;
    }

    template<typename ProposalImpl>
    std::vector<std::string> ReversibleJumpProposal<ProposalImpl>::getParameterNames() const {
        // Vector is constructed on demand, because it typically is not used repeatedly.
        std::vector<std::string> parameterNames = ProposalImpl::getParameterNames();
        std::vector<std::string> allParameterNames;

        // Loops over jumpable parameters and adds their activation state to the list of parameter names
        for (long i = 0; i < jumpIndices.rows(); ++i) {
            allParameterNames.emplace_back(
                    parameterNames.at(jumpIndices(i)) + "_active"
            );
        }

        allParameterNames.insert(allParameterNames.end(), parameterNames.begin(), parameterNames.end());
        return allParameterNames;
    }

    template<typename ProposalImpl>
    std::any ReversibleJumpProposal<ProposalImpl>::getParameter(const std::string &parameterName) const {
        return std::any();
    }

    template<typename ProposalImpl>
    std::string ReversibleJumpProposal<ProposalImpl>::getParameterType(const std::string &name) const {
        return std::string();
    }

    template<typename ProposalImpl>
    void ReversibleJumpProposal<ProposalImpl>::setParameter(const std::string &parameterName, const std::any &value) {

    }

    template<typename ProposalImpl>
    bool ReversibleJumpProposal<ProposalImpl>::hasStepSize() const {
        return true;
    }

    template<typename ProposalImpl>
    std::string ReversibleJumpProposal<ProposalImpl>::getProposalName() const {
        return "RJMCMC(" + ProposalImpl::getProposalName() + ")";
    }

    template<typename ProposalImpl>
    std::unique_ptr<Proposal> ReversibleJumpProposal<ProposalImpl>::copyProposal() const {
        return std::unique_ptr<ReversibleJumpProposal>(*this);
    }

    template<typename ProposalImpl>
    std::tuple<double, VectorType, VectorType>
    ReversibleJumpProposal<ProposalImpl>::jumpModel(RandomNumberGenerator &randomNumberGenerator,
                                                    const VectorType &activationState,
                                                    const VectorType &state,
                                                    const VectorType &defaultValues) {
        Eigen::VectorXi activationTracker = Eigen::VectorXi::Zero(defaultValues.rows());
        Eigen::VectorXi deactivationTracker = Eigen::VectorXi::Zero(defaultValues.rows());

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutationMatrix(jumpIndices_.size());
        permutationMatrix.setIdentity();
        std::shuffle(permutationMatrix.indices().data(),
                     permutationMatrix.indices().data() + permutationMatrix.indices().size(),
                     randomNumberGenerator);
        Eigen::VectorXi shuffledJumpIndices = permutationMatrix * jumpIndices_;
        auto A = Model::getA();
        auto b = Model::getB();
        double j_fwd = 1;
        double u_fwd = 1;
        std::vector<double> tester;
        for (long index = 0; index < shuffledJumpIndices.size(); ++index) {
            // If parameter is active, sample deactivation
            double defaultValue = defaultValues[shuffledJumpIndices(index)];
            if (proposalActivationState[shuffledJumpIndices(index)]) {
                // NOTE, WE MEASURE DISTANCE FROM DEFAULT BOTH IN ACTIVATION AND DEACTIVATION
                // FOR NON-SQUARE REGIONS, SWITCH OUT DEFAULT FOR EVERYTHING EXCEPT THE CURRENT INDEX
                auto[backwardsDistance, forwardsDistance] = distanceInCoordinateDirection(
                        A,
                        b,
                        defaultValues,
                        shuffledJumpIndices(index));
                double width = (forwardsDistance - backwardsDistance);
                // FOR NOW WE USE GAUSSIANS FOR BOTH ACTIVATION AND DEACTIVATION
                double temp = gaussianStepDistribution.computeProbabilityDensity(
                        proposal[shuffledJumpIndices(index)] - defaultValue, stepSize * width, backwardsDistance,
                        forwardsDistance);

                double procProb = std::min(temp * parameterDeactivationProbability, 1.);
                if (uniformRealDistribution(randomNumberGenerator) < procProb) {
                    proposalActivationState[shuffledJumpIndices(index)] = 0;
                    proposal[shuffledJumpIndices(index)] = defaultValue;
                    deactivationTracker(shuffledJumpIndices(index)) = 1;
                    j_fwd *= procProb;
                } else {
                    j_fwd *= 1 - procProb;
                }
            } else { // If parameter is inactive, sample activation
                if (uniformRealDistribution(randomNumberGenerator) < parameterActivationProbability) {
                    proposalActivationState[shuffledJumpIndices(index)] = 1;
                    // SAME AS ABOVE
                    auto[backwardsDistance, forwardsDistance] = distanceInCoordinateDirection(
                            A,
                            b,
                            defaultValues,
                            shuffledJumpIndices(index));

                    double width = (forwardsDistance - backwardsDistance);
                    // RESETTING THE BOUNDS OF THE GAUSSIAN STEP DISTRIBUTION IS A WASTE (DEPENDING ON GENERALITY), WE COULD INSTEAD SAVE
                    // ONE DISTRIBUTION PER JUMPSTATE
                    proposal[shuffledJumpIndices(index)] += gaussianStepDistribution.draw(randomNumberGenerator,
                                                                                          stepSize * width,
                                                                                          backwardsDistance,
                                                                                          forwardsDistance);
                    // What does this assert do?
                    assert(defaultValue + backwardsDistance <= proposal[shuffledJumpIndices(index)] &&
                           forwardsDistance + defaultValue >= proposal[shuffledJumpIndices(index)]);
                    activationTracker(shuffledJumpIndices(index)) = 1;
                    // GET THE DENSITY OF THE TAKEN STEP
                    double temp = gaussianStepDistribution.computeProbabilityDensity(
                            proposal[shuffledJumpIndices(index)] - defaultValue, stepSize * width,
                            backwardsDistance,
                            forwardsDistance);
                    tester.push_back(proposal[shuffledJumpIndices(index)] - defaultValue);
                    tester.push_back(stepSize * width);
                    tester.push_back(temp);
                    assert(temp > 0);
                    // HERE WE HAVE TO MULTIPLY WITH THE WIDTH TO COMPENSATE FOR THE PRIOR VOLUMES
                    u_fwd *= temp * width;
                    j_fwd *= parameterActivationProbability;
                } else {
                    j_fwd *= 1 - parameterActivationProbability;
                }
            }
        }

        double j_bck = 1;
        double u_bck = 1;
        std::shuffle(permutationMatrix.indices().data(),
                     permutationMatrix.indices().data() + permutationMatrix.indices().size(),
                     randomNumberGenerator);
        shuffledJumpIndices = permutationMatrix * jumpIndices_;

        // computes reverse probability (swap deactivation with activation)
        for (long index = 0; index < shuffledJumpIndices.size(); ++index) {
            // Note the if now checks for deactivated states
            if (!proposalActivationState[shuffledJumpIndices(index)]) {
                if (deactivationTracker(shuffledJumpIndices(index))) {
                    // AGAIN WASTEFUL COMPUTATION OF CONSTANT
                    auto[backwardsDistance, forwardsDistance] = distanceInCoordinateDirection(
                            A,
                            b,
                            defaultValues,
                            shuffledJumpIndices(index));
                    j_bck *= parameterActivationProbability;
                    double defaultValue = defaultValues[shuffledJumpIndices(index)];
                    // NOW TRACK PROBABILITY FROM DEFAULT TO PREVIOUS STATE, GET BETTER WAY OF RETRIEVING PAST STATE WITHOUT + 1!!!!
                    auto temp_old_state = this->getState()[shuffledJumpIndices(index) + 1];
                    assert(defaultValue + backwardsDistance <= temp_old_state &&
                           forwardsDistance + defaultValue >= temp_old_state);
                    // also here we have to take the width into account
                    double width = (forwardsDistance - backwardsDistance);

                    double temp = gaussianStepDistribution.computeProbabilityDensity(
                            this->getState()[shuffledJumpIndices(index) + 1] - defaultValue, stepSize * width,
                            backwardsDistance,
                            forwardsDistance);
                    //tester.push_back(temp);
                    if (temp <= 0) {
                        temp += 0;
                    }
                    //assert(temp > 0);
                    u_bck *= temp * width;
                } else {
                    j_bck *= 1 - parameterActivationProbability;
                }
            } else {
                double defaultValue = defaultParameterValues[shuffledJumpIndices(index)];
                //Eigen::VectorXd shiftedProposal = proposal;
                //shiftedProposal(shuffledJumpIndices(index)) -= defaultValue;
                //NOW CHECK PROBABILITY OF DEACTIVATING THE PROPOSAL
                auto[backwardsDistance, forwardsDistance] = distanceInCoordinateDirection(
                        A,
                        b,
                        defaultValues,
                        shuffledJumpIndices(index));
                double width = (forwardsDistance - backwardsDistance);
                //double defaultDistance = std::abs(proposal[shuffledJumpIndices(index)] - defaultValue);
                //defaultDistance = defaultDistance / width;

                //double prob = std::pow(defaultDistance, parameterDeactivationProbability);
                double temp = gaussianStepDistribution.computeProbabilityDensity(
                        proposal[shuffledJumpIndices(index)] - defaultValue, stepSize * width, backwardsDistance,
                        forwardsDistance);
                tester.push_back(proposal[shuffledJumpIndices(index)] - defaultValue);
                tester.push_back(stepSize * width);
                tester.push_back(temp);
                double prob = std::min(temp * parameterDeactivationProbability, 1.);
                if (activationTracker(shuffledJumpIndices(index))) {
                    j_bck *= prob;
                } else {
                    j_bck *= 1 - prob;
                }
            }
        }
        return std::log(j_bck * u_bck / (j_fwd * u_fwd));
    }
}

//    /**
//     * @tparam ProposalImpl is required to have a model mixed in already.
//     */
//    template<typename ProposalImpl>
//class ReversibleJumpProposal : public Proposal {
//    public:
//        ReversibleJumpProposal(const ProposalImpl &proposalImpl,
//                               Eigen::VectorXi jumpIndices,
//                               const VectorType &parameterDefaultValues);
//
//
//        // Old
//        void draw(RandomNumberGenerator &randomNumberGenerator);
//
//        void drawInModelSpace(RandomNumberGenerator &randomNumberGenerator,
//                              std::vector<int> &parameterActivationStates,
//                              const VectorType &defaultValues);
//
//        void drawInParameterSpace(RandomNumberGenerator &randomNumberGenerator);
//
//        double getAcceptanceRate() {
//            return static_cast<double>(numberOfAcceptedProposals) / numberOfProposals;
//        }
//
//        VectorType getState() {
//            VectorType parameterState = MarkovChainImpl::getState();
//            VectorType state(parameterState.rows() + 1);
//            long modelIndex = std::accumulate(parameterActivationStates_.begin(),
//                                              parameterActivationStates_.end(),
//                                              0,
//                                              [](unsigned int x, unsigned int y) { return (x << 1) + y; });
//            state << modelIndex, parameterState;
//            return state;
//        }
//
//        [[nodiscard]] std::vector<std::string> getStateNames() const {
//            // Vector is constructed on demand, because it typically is not used repeatedly.
//            std::vector<std::string> parameterNames = Model::getParameterNames();
//            std::vector<std::string> names = {"model index"};
//            names.insert(names.end(), parameterNames.begin(), parameterNames.end());
//            return names;
//        }
//
//    private:
//        double stateNegativeLogLikelihood = 0;
//        double proposalNegativeLogLikelihood = 0;
//
//        // fixed value from https://doi.org/10.1093/bioinformatics/btz500
//        VectorType::Scalar modelJumpProbability = 0.5;
//        VectorType::Scalar parameterActivationProbability = 0.1;
//        VectorType::Scalar parameterDeactivationProbability = 0.1;
//        VectorType::Scalar stepSize = 0.1;
//        std::uniform_real_distribution<double> uniformRealDistribution;
//        hops::GaussianStepDistribution<double> gaussianStepDistribution;
//
//        VectorType defaultParameterValues;
//        Eigen::VectorXi jumpIndices_;
//
//        std::vector<int> parameterActivationStates_;
//
//        long numberOfAcceptedProposals = 0;
//        long numberOfProposals = 0;
//
//        double jumpModel(RandomNumberGenerator &randomNumberGenerator,
//                         std::vector<int> &proposalActivationState,
//                         VectorType &proposal,
//                         const VectorType &defaultValues);
//    };
//
//    template<typename MarkovChainImpl, typename Model>
//    ReversibleJumpProposal<MarkovChainImpl, Model>::ReversibleJumpProposal(const MarkovChainImpl &markovChainImpl,
//                                                                           const Model &model,
//                                                                           Eigen::VectorXi jumpIndices,
//                                                                           const VectorType &parameterDefaultValues) :
//            MarkovChainImpl(markovChainImpl),
//            Model(model),
//            jumpIndices_(std::move(jumpIndices)),
//            defaultParameterValues(parameterDefaultValues) {
//        for (long i = 0; i < parameterDefaultValues.rows(); i++) {
//            parameterActivationStates_.emplace_back(1);
//        }
//        // Starts with all optional parameters deactivated
//        VectorType parameterState = MarkovChainImpl::getState();
//        for (long i = 0; i < jumpIndices_.rows(); ++i) {
//            parameterActivationStates_[jumpIndices_(i)] = 0;
//            parameterState[jumpIndices_(i)] = defaultParameterValues[jumpIndices_(i)];
//        }
//        MarkovChainImpl::setState(parameterState);
//        stateNegativeLogLikelihood = Model::computeNegativeLogLikelihood(parameterState);
//    }
//
//    template<typename MarkovChainImpl, typename Model>
//    void ReversibleJumpProposal<MarkovChainImpl, Model>::draw(RandomNumberGenerator &randomNumberGenerator) {
//        if (uniformRealDistribution(randomNumberGenerator) < modelJumpProbability) {
//            drawInModelSpace(randomNumberGenerator, parameterActivationStates_, defaultParameterValues);
//        } else {
//            drawInParameterSpace(randomNumberGenerator);
//        }
//    }
//
//    template<typename MarkovChainImpl, typename Model>
//    void ReversibleJumpProposal<MarkovChainImpl, Model>::drawInModelSpace(RandomNumberGenerator &randomNumberGenerator,
//                                                                          std::vector<int> &parameterActivationStates,
//                                                                          const VectorType &defaultValues) {
//    }
// TODO remove
//    template<typename MarkovChainImpl, typename Model>
//    void
//    ReversibleJumpProposal<MarkovChainImpl, Model>::drawInParameterSpace(RandomNumberGenerator &randomNumberGenerator) {
//        numberOfProposals++;
//        auto[acceptanceProbability, proposal] = MarkovChainImpl::propose(randomNumberGenerator,
//                                                                         parameterActivationStates_);
//        if (std::isfinite(acceptanceProbability)) {
//            proposalNegativeLogLikelihood = Model::computeNegativeLogLikelihood(proposal);
//            acceptanceProbability += stateNegativeLogLikelihood - proposalNegativeLogLikelihood;
//        }
//
//        double acceptanceChance = std::log(uniformRealDistribution(randomNumberGenerator));
//        if (acceptanceChance < acceptanceProbability) {
//            MarkovChainImpl::acceptProposal();
//            stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
//            numberOfAcceptedProposals++;
//        }
//
//    }
//

#endif //HOPS_REVERSIBLEJUMPPROPOSAL_HPP
