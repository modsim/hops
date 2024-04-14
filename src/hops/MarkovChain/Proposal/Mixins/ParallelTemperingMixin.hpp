#ifndef HOPS_PARALLELTEMPERINGMIXIN_HPP
#define HOPS_PARALLELTEMPERINGMIXIN_HPP

#include <memory>

#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {

    template<typename ProposalImpl, typename ParallelTemperingImpl>
    class ParallelTemperingMixin : public Proposal {
    public:
        ParallelTemperingMixin(const ProposalImpl &proposalImpl,
                               const ParallelTemperingImpl &parallelTemperingImpl,
                               int proposalStepsPerCommunication = 10,
                               int proposalStepsCounter = 0) :
                proposalImpl(proposalImpl),
                parallelTemperingImpl(parallelTemperingImpl),
                proposalStepsPerCommunication(proposalStepsPerCommunication),
                proposalStepsCounter(proposalStepsCounter) {
            proposal = proposalImpl.getProposal();
        }

        VectorType &propose(RandomNumberGenerator &rng) override {
            proposalStepsCounter = (proposalStepsCounter + 1) % proposalStepsPerCommunication;
            if (this->lastProposalWasParallelTemperingExchange()) {
                proposal = parallelTemperingImpl.proposeStateExchange(&proposalImpl);
                rng();
                std::cout << "proposed state exchange " << proposal.transpose() << std::endl;
                return proposal;
            } else {
                proposal = proposalImpl.propose(rng);
                return proposal;
            }
        }

        double computeLogAcceptanceProbability() override {
            if (this->lastProposalWasParallelTemperingExchange()) {
                return parallelTemperingImpl.computeAcceptanceProbability();
            } else {
                return proposalImpl.computeLogAcceptanceProbability();
            }
        }

        VectorType &acceptProposal() override {
            if (this->lastProposalWasParallelTemperingExchange()) {
                proposalImpl.setState(proposal);
                return proposal;
            }
            return proposalImpl.acceptProposal();
        }

        void setState(const VectorType &state) override {
            proposalImpl.setState(state);
        }

        [[nodiscard]] VectorType getState() const override {
            return proposalImpl.getState();
        }

        [[nodiscard]] VectorType getProposal() const {
            return proposal;
        }

        void setProposal(const VectorType &newProposal) override {
            proposal = newProposal;
            proposalImpl.setProposal(proposal);
        }

        void setDimensionNames(const std::vector<std::string> &names) override {
            return proposalImpl.setDimensionNames(names);
        }

        std::vector<std::string> getDimensionNames() const override {
            return proposalImpl.getDimensionNames();
        }

        std::vector<std::string> getParameterNames() const override {
            return proposalImpl.getParameterNames();
        }

        std::any getParameter(const ProposalParameter &parameter) const override {
            return proposalImpl.getParameter(parameter);
        }

        [[nodiscard]] std::string getParameterType(const ProposalParameter &parameter) const override {
            return proposalImpl.getParameterType(parameter);
        }

        void setParameter(const ProposalParameter &parameter, const std::any &value) override {
            proposalImpl.setParameter(parameter, value);
        }

        [[nodiscard]] std::string getProposalName() const override {
            return proposalImpl.getProposalName() + " " + parallelTemperingImpl.getName();
        }

        [[nodiscard]] const MatrixType &getA() const override {
            return proposalImpl.getA();
        }

        [[nodiscard]] const VectorType &getB() const override {
            return proposalImpl.getB();
        }

        [[nodiscard]] std::unique_ptr<Proposal> copyProposal() const override {
            return std::make_unique<ParallelTemperingMixin<ProposalImpl, ParallelTemperingImpl>>(
                    proposalImpl,
                    parallelTemperingImpl,
                    proposalStepsPerCommunication,
                    proposalStepsCounter);
        }


    private:
        [[nodiscard]] bool lastProposalWasParallelTemperingExchange() const {
            return proposalStepsCounter == 0;
        }

        ProposalImpl proposalImpl;
        ParallelTemperingImpl parallelTemperingImpl;
        int proposalStepsPerCommunication;
        int proposalStepsCounter = 0;
        VectorType proposal;
    };
}

#endif //HOPS_PARALLELTEMPERINGMIXIN_HPP
