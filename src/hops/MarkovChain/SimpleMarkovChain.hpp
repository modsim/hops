#ifndef HOPS_SIMPLEMARKOVCHAIN_HPP
#define HOPS_SIMPLEMARKOVCHAIN_HPP

#include <Eigen/Core>
#include <memory>
#include <random>
#include <vector>

#include "MarkovChainAttribute.hpp"

#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/Proposal/Proposal.hpp>
#include <hops/Model/Model.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Transformation/Transformation.hpp>
#include <hops/Utility/VectorType.hpp>

namespace hops {
    class SimpleMarkovChain : MarkovChain {
    public:
        SimpleMarkovChain(std::unique_ptr<Proposal> proposalDistribution, 
                    VectorType startingPoint,
                    std::unique_ptr<Model> model = std::unique_ptr<Model>(), 
                    std::unique_ptr<Transformation> transformation = std::unique_ptr<Transformation>()) : 
                proposalDistribution(std::move(proposalDistribution)),
                model(std::move(model)),
                transformation(std::move(transformation)),
                stateLogLikelihood(0) {
            if (model) {
                stateLogLikelihood = -model->computeNegativeLogLikelihood(transformation->revert(proposalDistribution->getState()));
            }
        }
                

        std::pair<double, VectorType> draw(RandomNumberGenerator& rng, long thinning = 1) override {
            VectorType newState = proposalDistribution->getState();
            for (long i = 0; i < thinning; ++i) {
                // get a new proposal and the proposals contribution to the log acceptance probability
                auto[logAcceptanceProbability, proposal] = proposalDistribution->propose(rng);

                // if the proposal knows the proposals negative log likelihood, retrieve it
                auto proposalNegativeLogLikelihood = proposalDistribution->getNegativeLogLikelihood();
                double proposalLogLikelihood;
                if (proposalNegativeLogLikelihood) {
                    proposalLogLikelihood = -proposalNegativeLogLikelihood.value();
                } else {
                    proposalLogLikelihood = -model.computeNegativeLogLikelihood(transformation->revert(proposal));
                }

                logAcceptanceProbability += proposalLogLikelihood - stateLogLikelihood;

                // do the metropolis filter            
                if (std::log(uniform(rng)) <= logAcceptanceProbability) { // acceptance
                    newState = proposalDistribution->acceptProposal();
                    stateLogLikelihood = proposalLogLikelihood;
                }
            }

            return transformation->apply(newState);
        }

    private:
        std::unique_ptr<Proposal> proposalDistribution;
        std::unique_ptr<Model> model;
        std::unique_ptr<Transformation> transformation;

        double stateLogLikelihood;

        std::uniform_real_distribution<double> uniform = std::uniform_real_distribution<double>(0, 1);
    };
}

#endif // HOPS_SIMPLEMARKOVCHAIN_HPP
