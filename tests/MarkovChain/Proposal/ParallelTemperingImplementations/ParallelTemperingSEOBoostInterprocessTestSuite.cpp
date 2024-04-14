#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ParallelTemperingSEOBoostInterprocessTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <random>

#include "hops/MarkovChain/ParallelTempering/Coldness.hpp"
#include "hops/MarkovChain/Proposal/ParallelTemperingImplementations/ParallelTemperingSEOBoostInterprocess.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

namespace {
    class ProposalMock {
    public:
        ProposalMock() {
            state = Eigen::VectorXd::Ones(2);
        }

        void draw(hops::RandomNumberGenerator &randomNumberGenerator) {
            for (long i = 0; i < state.rows(); ++i) {
                state(i) = uniformRealDistribution(randomNumberGenerator);
            }
        }

        hops::VectorType getState() {
            return state;
        }

        void setState(hops::VectorType newState) {
            state = std::move(newState);
        }

        double getStateNegativeLogLikelihood() {
            return state(0);
        }

    private:
        hops::VectorType state;
        std::uniform_real_distribution<double> uniformRealDistribution{0, 1000};
    };
}

BOOST_AUTO_TEST_SUITE(ParallelTemperingSEOBoostInterprocessTestSuite)

//    BOOST_AUTO_TEST_CASE(shouldProposeExchance_RngStateRemainsInSync) {
//        ProposalMock markovChainMock;
//        hops::Coldness mockWithColdness(markovChainMock);
//        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
//        hops::RandomNumberGenerator checkPoint = sharedRandomNumberGenerator;
//        hops::ParallelTempering parallelTempering(mockWithColdness,
//                                                  sharedRandomNumberGenerator,
//                                                  1);
//        MPI_Comm TEST_COMMUNICATOR;
//        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
//        int world_size;
//        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
//        int world_rank;
//        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);
//        MPI_Barrier(TEST_COMMUNICATOR);
//
//        while (!parallelTempering.shouldProposeExchange()) {}
//
//        double expectedRngState = sharedRandomNumberGenerator - checkPoint;
//        double actualRngState = expectedRngState;
//
//        int recvFromRank = (world_rank - 1) >= 0 ? world_rank - 1 : world_size - 1;
//        int sendToRank = (world_rank + 1) % (world_size);
//
//        MPI_Sendrecv_replace(&actualRngState, 1, MPI_DOUBLE, sendToRank, 0, recvFromRank, 0, TEST_COMMUNICATOR,
//                             MPI_STATUS_IGNORE);
//
//        BOOST_CHECK(expectedRngState == actualRngState);
//    }

    BOOST_AUTO_TEST_CASE(EvenAgreeOnExchangePair_test) {
        hops::RandomNumberGenerator syncRng(42);
        std::vector<hops::ParallelTemperingSEOBoostInterprocess> parallelTemperings;
        long n_chains = 10;

        for (long i = 0; i < n_chains; ++i) {
            parallelTemperings.push_back(hops::ParallelTemperingSEOBoostInterprocess(
                    syncRng,
                    n_chains,
                    i,
                    "test"));
        }

        long n_test_runs = 10;
        for (long i = 0; i < n_test_runs; ++i) {
            for (auto &pt: parallelTemperings) {
                int partner = pt.findPartnerForSwap();
                int chainIndex = pt.chainIndex;
                BOOST_CHECK(chainIndex>=0);
                BOOST_CHECK(chainIndex<n_chains);
                BOOST_CHECK(partner>=0);
                BOOST_CHECK(partner<n_chains);
                BOOST_CHECK(partner==chainIndex+1 || partner==chainIndex-1 || partner==n_chains-1 || partner==0);
            }
        }
    }

    BOOST_AUTO_TEST_CASE(OddAgreeOnExchangePair_test) {
        hops::RandomNumberGenerator syncRng(42);
        std::vector<hops::ParallelTemperingSEOBoostInterprocess> parallelTemperings;
        long n_chains = 9;

        for (long i = 0; i < n_chains; ++i) {
            parallelTemperings.push_back(hops::ParallelTemperingSEOBoostInterprocess(
                    syncRng,
                    n_chains,
                    i,
                    "test"));
        }

        long n_test_runs = 10;
        for (long i = 0; i < n_test_runs; ++i) {
            for (auto &pt: parallelTemperings) {
                int partner = pt.findPartnerForSwap();
                int chainIndex = pt.chainIndex;
                BOOST_CHECK(chainIndex>=0);
                BOOST_CHECK(chainIndex<n_chains);
                BOOST_CHECK(partner>=-1);
                BOOST_CHECK(partner<n_chains);
                BOOST_CHECK(partner==chainIndex+1 || partner==chainIndex-1 || partner==n_chains-1 || partner==0 || partner==-1);
            }
        }
    }


//    BOOST_AUTO_TEST_CASE(processesAgreeOnExchangeProbability_test) {
//        ProposalMock markovChainMock;
//        hops::Coldness mockWithColdness(markovChainMock);
//        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
//        hops::ParallelTempering parallelTempering(mockWithColdness,
//                                                  sharedRandomNumberGenerator,
//                                                  0.5);
//
//        MPI_Comm TEST_COMMUNICATOR;
//        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
//        int world_size;
//        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
//        int world_rank;
//        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);
//
//        MPI_Barrier(TEST_COMMUNICATOR);
//
//        for (int i = 0; i < 10; ++i) {
//            std::pair<int, int> exchangePair = parallelTempering.generateChainPairForExchangeProposal();
//            if (exchangePair.first == world_rank || exchangePair.second == world_rank) {
//                int otherChainRank = world_rank == exchangePair.first ? exchangePair.second : exchangePair.first;
//
//                MPI_Request request;
//
//                double actualAcceptanceProbability;
//                MPI_Irecv(&actualAcceptanceProbability, 1, MPI_DOUBLE, otherChainRank, 0,
//                          TEST_COMMUNICATOR, &request);
//
//                double expectedAcceptanceProbability = parallelTempering.computeExchangeAcceptanceProbability(
//                        otherChainRank);
//                double sendProbability = expectedAcceptanceProbability;
//                MPI_Send(&sendProbability, 1, MPI_DOUBLE, otherChainRank, 0,
//                         TEST_COMMUNICATOR);
//
//                MPI_Status status;
//                MPI_Wait(&request, &status);
//                BOOST_CHECK(actualAcceptanceProbability == expectedAcceptanceProbability);
//            }
//        }
//    }
//
//    BOOST_AUTO_TEST_CASE(processesCanExchangeState_test) {
//        ProposalMock markovChainMock;
//        hops::Coldness<ProposalMock> markovChainMockWithColdnessAttribute(markovChainMock, 0.5);
//        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
//        hops::ParallelTempering parallelTempering(markovChainMockWithColdnessAttribute,
//                                                  sharedRandomNumberGenerator,
//                                                  0.5);
//
//        MPI_Comm TEST_COMMUNICATOR;
//        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
//        int world_size;
//        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
//        int world_rank;
//        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);
//
//
//        MPI_Barrier(TEST_COMMUNICATOR);
//
//        for (int i = 0; i < 10; ++i) {
//            std::pair<int, int> exchangePair = parallelTempering.generateChainPairForExchangeProposal();
//            if (exchangePair.first == world_rank || exchangePair.second == world_rank) {
//                int otherChainRank = world_rank == exchangePair.first ? exchangePair.second : exchangePair.first;
//
//                Eigen::VectorXd actual = parallelTempering.getState();
//                MPI_Sendrecv_replace(actual.data(), actual.size(), MPI_DOUBLE, otherChainRank, 0, otherChainRank, 0,
//                                     TEST_COMMUNICATOR, MPI_STATUS_IGNORE);
//
//                parallelTempering.exchangeStates(otherChainRank);
//                Eigen::VectorXd expected = parallelTempering.getState();
//                BOOST_CHECK(actual == expected);
//            }
//        }
//    }
//
//    BOOST_AUTO_TEST_CASE(processesStayInSync_test) {
//        ProposalMock markovChainMock;
//        hops::Coldness<ProposalMock> markovChainMockWithColdnessAttribute(markovChainMock, 0.5);
//        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
//        hops::RandomNumberGenerator checkPoint = sharedRandomNumberGenerator;
//
//        hops::ParallelTempering parallelTempering(markovChainMockWithColdnessAttribute,
//                                                  sharedRandomNumberGenerator,
//                                                  0.5);
//
//        MPI_Comm TEST_COMMUNICATOR;
//        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
//        int world_size;
//        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
//        int world_rank;
//        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);
//
//        MPI_Barrier(TEST_COMMUNICATOR);
//        for (int i = 0; i < 10; ++i) {
//            parallelTempering.executeParallelTemperingStep();
//        }
//
//        double expectedRngState = sharedRandomNumberGenerator - checkPoint;
//        double actualRngState = expectedRngState;
//
//        int recvFromRank = (world_rank - 1) >= 0 ? world_rank - 1 : world_size - 1;
//        int sendToRank = (world_rank + 1) % (world_size);
//
//        MPI_Sendrecv_replace(&actualRngState, 1, MPI_DOUBLE, sendToRank, 0, recvFromRank, 0, TEST_COMMUNICATOR,
//                             MPI_STATUS_IGNORE);
//
//        BOOST_CHECK(expectedRngState == actualRngState);
//        MPI_Barrier(TEST_COMMUNICATOR);
//    }

BOOST_AUTO_TEST_SUITE_END()
