#ifndef HOPS_REVERSIBLEJUMPPROPOSALOLD_HPP
#define HOPS_REVERSIBLEJUMPPROPOSALOLD_HPP

#include <Eigen/Core>
#include <string>
#include <utility>
#include <vector>
#include <random>

#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/VectorType.hpp>

#include "Proposal.hpp"


namespace {
    // TODO move this function and also call it in CHRR...
    std::pair<double, double> distanceInCoordinateDirection(const Eigen::MatrixXd &A,
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

    Eigen::VectorXi
    shuffleJumpIndices(hops::RandomNumberGenerator &randomNumberGenerator, const Eigen::VectorXi &jumpIndices) {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutationMatrix(jumpIndices.size());
        permutationMatrix.setIdentity();
        std::shuffle(permutationMatrix.indices().data(),
                     permutationMatrix.indices().data() + permutationMatrix.indices().size(),
                     randomNumberGenerator);
        return permutationMatrix * jumpIndices;
    }

}

namespace hops {
    // TODO
    //  add proposal info to track whether it was model jump or parameter jump

    /**
     * @tparam ProposalImpl is required to have a model mixed in already.
     */
    template<typename ProposalImpl>
    class ReversibleJumpProposalOld : public ProposalImpl, public Proposal {
    public:
        ReversibleJumpProposalOld(const ProposalImpl &proposalImpl,
                                  Eigen::VectorXi jumpIndices,
                                  VectorType parameterDefaultValues);

        VectorType &propose(RandomNumberGenerator &rng) override;

        VectorType &proposeModel(RandomNumberGenerator &rng) override;

        VectorType &proposeParameters(RandomNumberGenerator &rng) override;

        double computeLogAcceptanceProbability() override;

        VectorType &acceptProposal() override;

        void setState(VectorType state) override;

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
        // TODO add these values to the parameters
        VectorType::Scalar modelJumpProbability = 0.5;
        VectorType::Scalar parameterActivationProbability = 0.1;
        VectorType::Scalar parameterDeactivationProbability = 0.1;
        // TODO rename stepsize because underlying proposal alreayd has stepSize
        VectorType::Scalar stepSize = 0.1;
        std::uniform_real_distribution<double> uniformRealDistribution;
        hops::GaussianStepDistribution<double> gaussianStepDistribution;

        Eigen::VectorXi jumpIndices;
        VectorType defaultValues;

        // The activation vectors also contain elements related to parameters, which are not jumped.
        std::vector<unsigned char> activationProposal;
        std::vector<unsigned char> activationState;

        double logProposalAcceptanceProbability;

        /**
         * @param randomNumberGenerator
         * @return modelJumpProb, proposal
         */
        std::tuple<double, VectorType> jumpModel(RandomNumberGenerator &randomNumberGenerator,
                                                 const VectorType &state);
    };

    template<typename ProposalImpl>
    ReversibleJumpProposalOld<ProposalImpl>::ReversibleJumpProposalOld(const ProposalImpl &proposalImpl,
                                                                       Eigen::VectorXi jumpIndices,
                                                                       VectorType parameterDefaultValues) :
            ProposalImpl(proposalImpl),
            jumpIndices(std::move(jumpIndices)),
            defaultValues(std::move(parameterDefaultValues)) {

        activationState = Eigen::VectorXi::Ones(defaultValues.rows());

        // Occam's Razor: Starts with all optional parameters deactivated.
        // TODO move construction of activation state out
        for (long i = 0; i < jumpIndices.rows(); ++i) {
            activationState(jumpIndices(i)) = 0;
        }

        activationProposal = activationState;
        ReversibleJumpProposalOld::setState(ProposalImpl::getState());

        // In case acceptance chance is computed before a proposal move has been made, return -inifinity to always reject.
        logProposalAcceptanceProbability = -std::numeric_limits<double>::infinity();
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposalOld<ProposalImpl>::propose(RandomNumberGenerator &rng) {
        if (uniformRealDistribution(rng) < modelJumpProbability) {
            return proposeModel(rng);
        } else {
            return proposeParameters(rng);
        }
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposalOld<ProposalImpl>::proposeParameters(RandomNumberGenerator &rng) {
        VectorType &proposal = ProposalImpl::propose(rng, activationState);
        logProposalAcceptanceProbability = ProposalImpl::computeLogAcceptanceProbability();
        return proposal;
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposalOld<ProposalImpl>::proposeModel(RandomNumberGenerator &rng) {
        VectorType state = ProposalImpl::getState();

        // TODO return activationProposal & logModelJumpProbability
        double logModelJumpProbabilityDifferential = jumpModel(rng, state);

        logProposalAcceptanceProbability = ProposalImpl::getStateNegativeLogLikelihood() -
                                           ProposalImpl::getProposalNegativeLogLikelihood() +
                                           logModelJumpProbabilityDifferential;

        // TODO ...
//        return proposal;
    }

    template<typename ProposalImpl>
    double ReversibleJumpProposalOld<ProposalImpl>::computeLogAcceptanceProbability() {
        return logProposalAcceptanceProbability;
    }

    template<typename ProposalImpl>
    VectorType &ReversibleJumpProposalOld<ProposalImpl>::acceptProposal() {
        ProposalImpl::acceptProposal();
        activationState = activationProposal;
        return ProposalImpl::getState();
    }

    template<typename ProposalImpl>
    void ReversibleJumpProposalOld<ProposalImpl>::setState(VectorType newState) {
        for (long i = 0; i < jumpIndices.rows(); ++i) {
            newState[jumpIndices(i)] = defaultValues[jumpIndices(i)];
        }

        ProposalImpl::setState(newState);
    }

    template<typename ProposalImpl>
    VectorType ReversibleJumpProposalOld<ProposalImpl>::getState() const {
        return ProposalImpl::getState();
    }

    template<typename ProposalImpl>
    VectorType ReversibleJumpProposalOld<ProposalImpl>::getProposal() const {
        return ProposalImpl::getProposal();
    }

    template<typename ProposalImpl>
    std::vector<std::string> ReversibleJumpProposalOld<ProposalImpl>::getParameterNames() const {
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
    std::any ReversibleJumpProposalOld<ProposalImpl>::getParameter(const std::string &parameterName) const {
        // TODO
        return std::any();
    }

    template<typename ProposalImpl>
    std::string ReversibleJumpProposalOld<ProposalImpl>::getParameterType(const std::string &name) const {
        // TODO
        return std::string();
    }

    template<typename ProposalImpl>
    void ReversibleJumpProposalOld<ProposalImpl>::setParameter(const std::string &parameterName, const std::any &value) {
        // TODO
    }

    template<typename ProposalImpl>
    bool ReversibleJumpProposalOld<ProposalImpl>::hasStepSize() const {
        return true;
    }

    template<typename ProposalImpl>
    std::string ReversibleJumpProposalOld<ProposalImpl>::getProposalName() const {
        return "RJMCMC(" + ProposalImpl::getProposalName() + ")";
    }

    template<typename ProposalImpl>
    std::unique_ptr<Proposal> ReversibleJumpProposalOld<ProposalImpl>::copyProposal() const {
        return std::unique_ptr<ReversibleJumpProposalOld>(*this);
    }

    template<typename ProposalImpl>
    std::tuple<double, VectorType>
    ReversibleJumpProposalOld<ProposalImpl>::jumpModel(RandomNumberGenerator &randomNumberGenerator,
                                                       const VectorType &state) {
        VectorType proposal = state;
        Eigen::VectorXi activationTracker = Eigen::VectorXi::Zero(defaultValues.rows());
        Eigen::VectorXi deactivationTracker = Eigen::VectorXi::Zero(defaultValues.rows());

        Eigen::VectorXi shuffledJumpIndices = shuffleJumpIndices(randomNumberGenerator, jumpIndices);
        auto A = ProposalImpl::getA();
        auto b = ProposalImpl::getB();

        double j_fwd = 1;
        double u_fwd = 1;
        double j_bck = 1;
        double u_bck = 1;

        std::vector<double> tester;
        for (long index = 0; index < shuffledJumpIndices.size(); ++index) {
            // If parameter is active, sample deactivation
            if (activationState[shuffledJumpIndices(index)]) {
                // NOTE, WE MEASURE DISTANCE FROM DEFAULT BOTH IN ACTIVATION AND DEACTIVATION
                // FOR NON-SQUARE REGIONS, SWITCH OUT DEFAULT FOR EVERYTHING EXCEPT THE CURRENT INDEX
                auto[backwardsDistance, forwardsDistance] = distanceInCoordinateDirection(
                        A,
                        b,
                        defaultValues,
                        shuffledJumpIndices(index));
                double width = (forwardsDistance - backwardsDistance);
                double defaultValue = defaultValues[shuffledJumpIndices(index)];
                // FOR NOW WE USE GAUSSIANS FOR BOTH ACTIVATION AND DEACTIVATION
                double temp = gaussianStepDistribution.computeProbabilityDensity(
                        proposal[shuffledJumpIndices(index)] - defaultValue, stepSize * width, backwardsDistance,
                        forwardsDistance);

                double procProb = std::min(temp * parameterDeactivationProbability, 1.);
                if (uniformRealDistribution(randomNumberGenerator) < procProb) {
                    activationProposal[shuffledJumpIndices(index)] = 0;
                    proposal[shuffledJumpIndices(index)] = defaultValue;
                    deactivationTracker(shuffledJumpIndices(index)) = 1;
                    j_fwd *= procProb;
                } else {
                    j_fwd *= 1 - procProb;
                }
            } else { // If parameter is inactive, sample activation
                if (uniformRealDistribution(randomNumberGenerator) < parameterActivationProbability) {
                    activationProposal[shuffledJumpIndices(index)] = 1;
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
//                    // What does this assert do?
//                    assert(defaultValue + backwardsDistance <= proposal[shuffledJumpIndices(index)] &&
//                           forwardsDistance + defaultValue >= proposal[shuffledJumpIndices(index)]);
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


        // Why the reshuffle?!
        std::shuffle(permutationMatrix.indices().data(),
                     permutationMatrix.indices().data() + permutationMatrix.indices().size(),
                     randomNumberGenerator);
        shuffledJumpIndices = permutationMatrix * jumpIndices;

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
                double defaultValue = defaultValues[shuffledJumpIndices(index)];
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

    template<typename ProposalImpl>
    std::tuple<double, VectorType, VectorType>
    ReversibleJumpProposalOld<ProposalImpl>::jumpModel(RandomNumberGenerator &randomNumberGenerator,
                                                       const VectorType &state) {
        Eigen::VectorXi activationTracker = Eigen::VectorXi::Zero(defaultValues.rows());
        Eigen::VectorXi deactivationTracker = Eigen::VectorXi::Zero(defaultValues.rows());

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutationMatrix(jumpIndices.size());
        permutationMatrix.setIdentity();
        std::shuffle(permutationMatrix.indices().data(),
                     permutationMatrix.indices().data() + permutationMatrix.indices().size(),
                     randomNumberGenerator);
        Eigen::VectorXi shuffledJumpIndices = permutationMatrix * jumpIndices;
        auto A = ProposalImpl::getA();
        auto b = ProposalImpl::getB();
        double j_fwd = 1;
        double u_fwd = 1;
        std::vector<double> tester;
        for (long index = 0; index < shuffledJumpIndices.size(); ++index) {
            // If parameter is active, sample deactivation
            if (activationState[shuffledJumpIndices(index)]) {
                // NOTE, WE MEASURE DISTANCE FROM DEFAULT BOTH IN ACTIVATION AND DEACTIVATION
                // FOR NON-SQUARE REGIONS, SWITCH OUT DEFAULT FOR EVERYTHING EXCEPT THE CURRENT INDEX
                auto[backwardsDistance, forwardsDistance] = distanceInCoordinateDirection(
                        A,
                        b,
                        defaultValues,
                        shuffledJumpIndices(index));
                double width = (forwardsDistance - backwardsDistance);
                double defaultValue = defaultValues[shuffledJumpIndices(index)];
                // FOR NOW WE USE GAUSSIANS FOR BOTH ACTIVATION AND DEACTIVATION
                double temp = gaussianStepDistribution.computeProbabilityDensity(
                        proposal[shuffledJumpIndices(index)] - defaultValue, stepSize * width, backwardsDistance,
                        forwardsDistance);

                double procProb = std::min(temp * parameterDeactivationProbability, 1.);
                if (uniformRealDistribution(randomNumberGenerator) < procProb) {
                    activationProposal[shuffledJumpIndices(index)] = 0;
                    proposal[shuffledJumpIndices(index)] = defaultValue;
                    deactivationTracker(shuffledJumpIndices(index)) = 1;
                    j_fwd *= procProb;
                } else {
                    j_fwd *= 1 - procProb;
                }
            } else { // If parameter is inactive, sample activation
                if (uniformRealDistribution(randomNumberGenerator) < parameterActivationProbability) {
                    activationProposal[shuffledJumpIndices(index)] = 1;
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
//                    // What does this assert do?
//                    assert(defaultValue + backwardsDistance <= proposal[shuffledJumpIndices(index)] &&
//                           forwardsDistance + defaultValue >= proposal[shuffledJumpIndices(index)]);
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
        shuffledJumpIndices = permutationMatrix * jumpIndices;

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
                double defaultValue = defaultValues[shuffledJumpIndices(index)];
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

#endif //HOPS_REVERSIBLEJUMPPROPOSALOLD_HPP
