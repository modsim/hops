#ifndef HOPS_PARALLELTEMPERING_HPP
#define HOPS_PARALLELTEMPERING_HPP

namespace hops {
    class Proposal;

    class RandomNumberGenerator;

    class VectorType;

    class Proposal;

    class ParallelTempering {
    public:
        /**
         * Communicates with other parallel chains (e.g. MPI or shared memory) and fetches a proposal for a new state
         * @param rng
         * @param state
         * @return proposal for state exchange
         */
        virtual VectorType proposeStateExchange(RandomNumberGenerator &rng, Proposal*) = 0;

        virtual double computeAcceptanceProbability() = 0;

        virtual ~ParallelTempering() = default;
    };
}

#endif //HOPS_PARALLELTEMPERING_HPP
