#define BOOST_TEST_MODULE ParallelTemperingTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/MarkovChain/ParallelTempering/ParallelTempering.hpp>
#include <hops/MarkovChain/ParallelTempering/ColdnessAttribute.hpp>

namespace {
    class MarkovChainMock {
    public:
        MarkovChainMock() {
            state = Eigen::VectorXd::Ones(2);
        }

        using StateType = Eigen::VectorXd;
        using MatrixType = Eigen::VectorXd;
        using VectorType = StateType;

        void draw(hops::RandomNumberGenerator &randomNumberGenerator) {
            for (long i = 0; i < state.rows(); ++i) {
                state(i) = uniformRealDistribution(randomNumberGenerator);
            }
        }

        StateType getState() {
            return state;
        }

        void setState(StateType newState) {
            state = std::move(newState);
        }

        double getNegativeLogLikelihoodOfCurrentState() {
            return state(0);
        }

    private:
        StateType state;
        std::uniform_real_distribution<double> uniformRealDistribution{0, 1000};
    };
}

BOOST_AUTO_TEST_SUITE(ParallelTempering)

    BOOST_AUTO_TEST_CASE(assertCorrectNumberOfProcessesAreRun) {
        int expectedNumberOfProcesses = 3; // Defined in CMakeLists.txt
        MarkovChainMock markovChainMock;
        hops::ColdnessAttribute mockWithColdness(markovChainMock);
        hops::ParallelTempering parallelTempering(mockWithColdness,
                                                  hops::RandomNumberGenerator(0),
                                                  0.00005);
        MPI_Comm TEST_COMMUNICATOR;
        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
        int world_size;
        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
        BOOST_CHECK(world_size == expectedNumberOfProcesses);
    }

    BOOST_AUTO_TEST_CASE(shouldSetChainTemperaturesCorrectly) {
        MarkovChainMock markovChainMock;
        hops::ColdnessAttribute mockWithColdness(markovChainMock);
        hops::ParallelTempering parallelTempering(mockWithColdness,
                                                  hops::RandomNumberGenerator(0),
                                                  0.00005);
        MPI_Comm TEST_COMMUNICATOR;
        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
        int size;
        MPI_Comm_size(TEST_COMMUNICATOR, &size);
        int rank;
        MPI_Comm_rank(TEST_COMMUNICATOR, &rank);
        BOOST_CHECK(parallelTempering.getColdness() == 1 - static_cast<double>(rank) / (size - 1));
    }

    BOOST_AUTO_TEST_CASE(shouldProposeExchance_RngStateRemainsInSync) {
        MarkovChainMock markovChainMock;
        hops::ColdnessAttribute mockWithColdness(markovChainMock);
        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
        hops::RandomNumberGenerator checkPoint = sharedRandomNumberGenerator;
        hops::ParallelTempering parallelTempering(mockWithColdness,
                                                  sharedRandomNumberGenerator,
                                                  1);
        MPI_Comm TEST_COMMUNICATOR;
        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
        int world_size;
        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
        int world_rank;
        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);
        MPI_Barrier(TEST_COMMUNICATOR);

        while (!parallelTempering.shouldProposeExchange()) {}

        double expectedRngState = sharedRandomNumberGenerator - checkPoint;
        double actualRngState = expectedRngState;

        int recvFromRank = (world_rank - 1) >= 0 ? world_rank - 1 : world_size - 1;
        int sendToRank = (world_rank + 1) % (world_size);

        MPI_Sendrecv_replace(&actualRngState, 1, MPI_DOUBLE, sendToRank, 0, recvFromRank, 0, TEST_COMMUNICATOR,
                             MPI_STATUS_IGNORE);

        BOOST_CHECK(expectedRngState == actualRngState);
    }

    BOOST_AUTO_TEST_CASE(processesAgreeOnExchangePair_test) {
        MarkovChainMock markovChainMock;
        hops::ColdnessAttribute mockWithColdness(markovChainMock);
        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
        hops::ParallelTempering parallelTempering(mockWithColdness,
                                                  sharedRandomNumberGenerator,
                                                  0.05);

        MPI_Comm TEST_COMMUNICATOR;
        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
        int world_size;
        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
        int world_rank;
        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);
        MPI_Barrier(TEST_COMMUNICATOR);


        MPI_Barrier(TEST_COMMUNICATOR);
        for (int i = 0; i < 10; ++i) {
            std::pair<int, int> exchangePair = parallelTempering.generateChainPairForExchangeProposal();
            int rngStateSend[] = {exchangePair.first, exchangePair.second};
            int rngStateRecv[2];
            int recvFromRank = (world_rank - 1) >= 0 ? world_rank - 1 : world_size - 1;
            int sendToRank = (world_rank + 1) % (world_size);
            MPI_Barrier(TEST_COMMUNICATOR);
            MPI_Barrier(TEST_COMMUNICATOR);

            MPI_Request request;
            MPI_Irecv(&rngStateRecv, 2, MPI_INT, recvFromRank, 0, TEST_COMMUNICATOR, &request);
            MPI_Send(&rngStateSend, 2, MPI_INT, sendToRank, 0, TEST_COMMUNICATOR);
            MPI_Status status;
            MPI_Wait(&request, &status);
            BOOST_CHECK(rngStateSend[0] == rngStateRecv[0]);
            BOOST_CHECK(rngStateSend[1] == rngStateRecv[1]);
        }
    }

    BOOST_AUTO_TEST_CASE(processesAgreeOnExchangeProbability_test) {
        MarkovChainMock markovChainMock;
        hops::ColdnessAttribute mockWithColdness(markovChainMock);
        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
        hops::ParallelTempering parallelTempering(mockWithColdness,
                                                  sharedRandomNumberGenerator,
                                                  0.5);

        MPI_Comm TEST_COMMUNICATOR;
        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
        int world_size;
        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
        int world_rank;
        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);

        MPI_Barrier(TEST_COMMUNICATOR);

        for (int i = 0; i < 10; ++i) {
            std::pair<int, int> exchangePair = parallelTempering.generateChainPairForExchangeProposal();
            if (exchangePair.first == world_rank || exchangePair.second == world_rank) {
                int otherChainRank = world_rank == exchangePair.first ? exchangePair.second : exchangePair.first;

                MPI_Request request;

                double actualAcceptanceProbability;
                MPI_Irecv(&actualAcceptanceProbability, 1, MPI_DOUBLE, otherChainRank, 0,
                          TEST_COMMUNICATOR, &request);

                double expectedAcceptanceProbability = parallelTempering.calculateExchangeAcceptanceProbability(
                        otherChainRank);
                double sendProbability = expectedAcceptanceProbability;
                MPI_Send(&sendProbability, 1, MPI_DOUBLE, otherChainRank, 0,
                         TEST_COMMUNICATOR);

                MPI_Status status;
                MPI_Wait(&request, &status);
                BOOST_CHECK(actualAcceptanceProbability == expectedAcceptanceProbability);
            }
        }
    }

    BOOST_AUTO_TEST_CASE(processesCanExchangeState_test) {
        MarkovChainMock markovChainMock;
        hops::ColdnessAttribute<MarkovChainMock> markovChainMockWithColdnessAttribute(markovChainMock, 0.5);
        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
        hops::ParallelTempering parallelTempering(markovChainMockWithColdnessAttribute,
                                                  sharedRandomNumberGenerator,
                                                  0.5);

        MPI_Comm TEST_COMMUNICATOR;
        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
        int world_size;
        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
        int world_rank;
        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);


        MPI_Barrier(TEST_COMMUNICATOR);

        for (int i = 0; i < 10; ++i) {
            std::pair<int, int> exchangePair = parallelTempering.generateChainPairForExchangeProposal();
            if (exchangePair.first == world_rank || exchangePair.second == world_rank) {
                int otherChainRank = world_rank == exchangePair.first ? exchangePair.second : exchangePair.first;

                Eigen::VectorXd actual = parallelTempering.getState();
                MPI_Sendrecv_replace(actual.data(), actual.size(), MPI_DOUBLE, otherChainRank, 0, otherChainRank, 0,
                                     TEST_COMMUNICATOR, MPI_STATUS_IGNORE);

                parallelTempering.exchangeStates(otherChainRank);
                Eigen::VectorXd expected = parallelTempering.getState();
                BOOST_CHECK(actual == expected);
            }
        }
    }

    BOOST_AUTO_TEST_CASE(processesStayInSync_test) {
        MarkovChainMock markovChainMock;
        hops::ColdnessAttribute<MarkovChainMock> markovChainMockWithColdnessAttribute(markovChainMock, 0.5);
        hops::RandomNumberGenerator sharedRandomNumberGenerator(42);
        hops::RandomNumberGenerator checkPoint = sharedRandomNumberGenerator;

        hops::ParallelTempering parallelTempering(markovChainMockWithColdnessAttribute,
                                                  sharedRandomNumberGenerator,
                                                  0.5);

        MPI_Comm TEST_COMMUNICATOR;
        MPI_Comm_dup(MPI_COMM_WORLD, &TEST_COMMUNICATOR);
        int world_size;
        MPI_Comm_size(TEST_COMMUNICATOR, &world_size);
        int world_rank;
        MPI_Comm_rank(TEST_COMMUNICATOR, &world_rank);

        MPI_Barrier(TEST_COMMUNICATOR);
        for (int i = 0; i < 10; ++i) {
            parallelTempering.executeParallelTemperingStep();
        }

        double expectedRngState = sharedRandomNumberGenerator - checkPoint;
        double actualRngState = expectedRngState;

        int recvFromRank = (world_rank - 1) >= 0 ? world_rank - 1 : world_size - 1;
        int sendToRank = (world_rank + 1) % (world_size);

        MPI_Sendrecv_replace(&actualRngState, 1, MPI_DOUBLE, sendToRank, 0, recvFromRank, 0, TEST_COMMUNICATOR,
                             MPI_STATUS_IGNORE);

        BOOST_CHECK(expectedRngState == actualRngState);
        MPI_Barrier(TEST_COMMUNICATOR);
    }

BOOST_AUTO_TEST_SUITE_END()