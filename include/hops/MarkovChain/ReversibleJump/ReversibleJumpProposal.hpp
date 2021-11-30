#ifndef HOPS_REVERSIBLEJUMPPROPOSAL_HPP
#define HOPS_REVERSIBLEJUMPPROPOSAL_HPP

#include <Eigen/Core>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
    std::pair<double, double>
    distanceInCoordinateDirection(const Eigen::MatrixXd &A,
                                  const Eigen::VectorXd &b,
                                  const Eigen::VectorXd &x,
                                  long coordinate) {
        std::cout << "x " << x.transpose() << std::endl;
        Eigen::VectorXd slacks = b - A * x;
        Eigen::VectorXd inverseDistances = A.col(coordinate).cwiseQuotient(slacks);
        double forwardDistance = 1. / inverseDistances.maxCoeff();
        double backwardDistance = 1. / inverseDistances.minCoeff();

        for (long i = 0; i < inverseDistances.rows(); ++i) {
            if (inverseDistances(i) == -std::numeric_limits<double>::infinity()) {
                backwardDistance=0;
            } else if (inverseDistances(i) == std::numeric_limits<double>::infinity()) {
                forwardDistance=0;
            }
        }
        assert(backwardDistance <= 0 && forwardDistance >= 0);
        return std::make_pair(backwardDistance, forwardDistance);
    }
}

namespace hops {
    template<typename MarkovChainImpl, typename Model>
    class ReversibleJumpProposal : public MarkovChainImpl, public Model {
    public:
        ReversibleJumpProposal(const MarkovChainImpl &markovChainImpl,
                               const Model &model,
                               Eigen::VectorXi jumpIndices,
                               const typename MarkovChainImpl::StateType parameterDefaultValues) :
                MarkovChainImpl(markovChainImpl),
                Model(model),
                jumpIndices_(std::move(jumpIndices)),
                defaultParameterValues(parameterDefaultValues) {
            for (long i = 0; i < parameterDefaultValues.rows(); i++) {
                parameterActivationStates_.emplace_back(1);
            }
            // Starts with all optional parameters deactivated
            typename MarkovChainImpl::StateType parameterState = MarkovChainImpl::getState();
            for (long i = 0; i < jumpIndices_.rows(); ++i) {
                parameterActivationStates_[jumpIndices_(i)] = 0;
                parameterState[jumpIndices_(i)] = defaultParameterValues[jumpIndices_(i)];
            }
            MarkovChainImpl::setState(parameterState);
        }

        void draw(RandomNumberGenerator &randomNumberGenerator) {
            if (uniformRealDistribution(randomNumberGenerator) < modelJumpProbability) {
                drawInModelSpace(randomNumberGenerator, parameterActivationStates_, defaultParameterValues);
            } else {
                drawInParameterSpace(randomNumberGenerator);
            }
        }

        void drawInModelSpace(RandomNumberGenerator &randomNumberGenerator,
                              std::vector<int> &parameterActivationStates,
                              typename MarkovChainImpl::StateType defaultValues) {
            typename MarkovChainImpl::StateType proposal = MarkovChainImpl::getState();
            std::vector<int> proposalActivationStates(parameterActivationStates_);

            double logModelJumpProbabilityDifferential = jumpModel(randomNumberGenerator,
                                                                   proposalActivationStates,
                                                                   proposal,
                                                                   defaultValues);

            proposalNegativeLogLikelihood = Model::computeNegativeLogLikelihood(proposal);
            double modelJumpAcceptanceProbability = stateNegativeLogLikelihood - proposalNegativeLogLikelihood +
                                                    logModelJumpProbabilityDifferential;
            if (std::log(uniformRealDistribution(randomNumberGenerator)) <= modelJumpAcceptanceProbability) {
                stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
                proposalActivationStates.swap(parameterActivationStates_);
                MarkovChainImpl::setState(proposal);
            } else {
                // TODO update rejection stats
            }
        }


        void drawInParameterSpace(RandomNumberGenerator &randomNumberGenerator) {
            numberOfProposals++;
            MarkovChainImpl::propose(randomNumberGenerator);
            for (long i = 0; i < MarkovChainImpl::proposal.rows(); ++i) {
                if (!parameterActivationStates_[i]) {
                    // If parameter is not active, reset proposal to state
                    MarkovChainImpl::proposal(i) = MarkovChainImpl::state(i);
                }
            }
            double acceptanceProbability = computeParameterDrawAcceptanceProbability();
            double acceptanceChance = std::log(uniformRealDistribution(randomNumberGenerator));
            if (acceptanceChance < acceptanceProbability) {
                MarkovChainImpl::acceptProposal();
                stateNegativeLogLikelihood = proposalNegativeLogLikelihood;
                numberOfAcceptedProposals++;
            }
        }

        double computeParameterDrawAcceptanceProbability() {
            double acceptanceProbability = 0;
            if constexpr(IsCalculateLogAcceptanceProbabilityAvailable<MarkovChainImpl>::value) {
                acceptanceProbability += MarkovChainImpl::computeLogAcceptanceProbability();
            }
            if (std::isfinite(acceptanceProbability)) {
                proposalNegativeLogLikelihood = Model::computeNegativeLogLikelihood(
                        MarkovChainImpl::getProposal());
                acceptanceProbability += stateNegativeLogLikelihood - proposalNegativeLogLikelihood;
            }
            return acceptanceProbability;
        }

        double getAcceptanceRate() {
            return static_cast<double>(numberOfAcceptedProposals) / numberOfProposals;
        }

        typename MarkovChainImpl::StateType getState() {
            typename MarkovChainImpl::StateType parameterState = MarkovChainImpl::getState();
            typename MarkovChainImpl::StateType state(parameterState.rows() + 1);
            long modelIndex = std::accumulate(parameterActivationStates_.begin(),
                                              parameterActivationStates_.end(),
                                              0,
                                              [](unsigned int x, unsigned int y) { return (x << 1) + y; });
            state << modelIndex, parameterState;
            return state;
        }

        [[nodiscard]] std::vector<std::string> getParameterNames() const {
            // Vector is constructed on demand, because it typically is not used repeatedly.
            std::vector<std::string> parameterNames = Model::getParameterNames();
            std::vector<std::string> names = {"model index"};
            names.insert(names.end(), parameterNames.begin(), parameterNames.end());
            return names;
        }

    private:
        double stateNegativeLogLikelihood = 0;
        double proposalNegativeLogLikelihood = 0;

        // fixed value from https://doi.org/10.1093/bioinformatics/btz500
        typename MarkovChainImpl::StateType::Scalar modelJumpProbability = 0.5;
        typename MarkovChainImpl::StateType::Scalar parameterActivationProbability = 0.1;
        typename MarkovChainImpl::StateType::Scalar parameterDeactivationProbability = 0.1;
        typename MarkovChainImpl::StateType::Scalar stepSize = 0.1;
        std::uniform_real_distribution<double> uniformRealDistribution;
        hops::GaussianStepDistribution<double> gaussianStepDistribution;

        typename MarkovChainImpl::StateType defaultParameterValues;
        Eigen::VectorXi jumpIndices_;

        std::vector<int> parameterActivationStates_;

        long numberOfAcceptedProposals = 0;
        long numberOfProposals = 0;


        double jumpModel(hops::RandomNumberGenerator &randomNumberGenerator,
                         std::vector<int> &proposalActivationState,
                         typename MarkovChainImpl::StateType &proposal,
                         typename MarkovChainImpl::StateType &defaultValues) {
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
    };
}

#endif //HOPS_REVERSIBLEJUMPPROPOSAL_HPP
