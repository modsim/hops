#define BOOST_TEST_MODULE PotentialScaleReductionFactorTestSuite

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <hops/hops.hpp>

BOOST_AUTO_TEST_SUITE(PotentialScaleReductionFactorTestSuite)
    BOOST_AUTO_TEST_CASE(ComputeAllDraws) {
        // Expected results were evaluated using stan implementation
        size_t numChains = 4;
        auto expectedResult = std::vector<double>{3.87441, 4.94213, 3.52473, 6.43039, 4.06456};
        std::vector<std::vector<Eigen::VectorXd>> chains(numChains);
        for (size_t i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

            for (long j = 0; j < statesMatrix.rows(); ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains[i] = draws;
        }

        std::vector<double> actualResult = hops::computePotentialScaleReductionFactor(chains);

        for (size_t i = 0; i < expectedResult.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult[i], actualResult[i], 0.01);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeAllDrawsPointerArray) {
        // Expected results were evaluated using stan implementation
        size_t numChains = 4;
        auto expectedResult = std::vector<double>{3.87441, 4.94213, 3.52473, 6.43039, 4.06456};
        std::vector<std::vector<Eigen::VectorXd>> chains(numChains);
        std::vector<const std::vector<Eigen::VectorXd>*> chainsPtrArray(numChains);
        for (size_t i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

            for (long j = 0; j < statesMatrix.rows(); ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains[i] = draws;
            chainsPtrArray[i] = &chains[i];
        }

        std::vector<double> actualResult = hops::computePotentialScaleReductionFactor(chains);

        for (size_t i = 0; i < expectedResult.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult[i], actualResult[i], 0.01);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeIncrementally) {
        // Expected results were evaluated using stan implementation
        size_t numChains = 4;
        auto expectedResult = std::vector<double>{8.18767, 7.33072, 5.33744, 9.95099, 6.34195};
        std::vector<std::vector<Eigen::VectorXd>> chains(numChains);
        for (size_t i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            std::vector<Eigen::VectorXd> draws(static_cast<unsigned long>(statesMatrix.rows() / 2));

            for (long j = 0; j < statesMatrix.rows() / 2; ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains[i] = draws;
        }

        unsigned long numUnseen = static_cast<unsigned long>(chains[0].size());
        std::vector<std::vector<double>> sampleVariances;
        std::vector<std::vector<double>> intraChainExpectations;
        std::vector<double> interChainExpectation;
        unsigned long numSeen = 0;

        std::vector<double> actualResult = hops::computePotentialScaleReductionFactor<Eigen::VectorXd>(
                chains, numUnseen, sampleVariances, intraChainExpectations, interChainExpectation, numSeen);

        for (size_t i = 0; i < expectedResult.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult[i], actualResult[i], 0.01);
        }

        expectedResult = std::vector<double>{3.87441, 4.94213, 3.52473, 6.43039, 4.06456};
        for (size_t i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            for (long j = statesMatrix.rows() / 2; j < statesMatrix.rows(); ++j) {
                chains[i].push_back(statesMatrix.row(j));
            }
        }

        actualResult = hops::computePotentialScaleReductionFactor(
                chains, numUnseen, sampleVariances, intraChainExpectations, interChainExpectation, numSeen);

        for (size_t i = 0; i < expectedResult.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult[i], actualResult[i], 0.01);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeIncrementallyPointerArray) {
        // Expected results were evaluated using stan implementation
        size_t numChains = 4;
        auto expectedResult = std::vector<double>{8.18767, 7.33072, 5.33744, 9.95099, 6.34195};
        std::vector<std::vector<Eigen::VectorXd>> chains(numChains);
        std::vector<const std::vector<Eigen::VectorXd>*> chainsPtrArray(numChains);
        for (size_t i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            std::vector<Eigen::VectorXd> draws(static_cast<unsigned long>(statesMatrix.rows() / 2));

            for (long j = 0; j < statesMatrix.rows() / 2; ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains[i] = draws;
            chainsPtrArray[i] = &chains[i];
        }

        unsigned long numUnseen = static_cast<unsigned long>(chains[0].size());
        std::vector<std::vector<double>> sampleVariances;
        std::vector<std::vector<double>> intraChainExpectations;
        std::vector<double> interChainExpectation;
        unsigned long numSeen = 0;

        std::vector<double> actualResult = hops::computePotentialScaleReductionFactor<Eigen::VectorXd>(
                chains, numUnseen, sampleVariances, intraChainExpectations, interChainExpectation, numSeen);

        for (size_t i = 0; i < expectedResult.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult[i], actualResult[i], 0.01);
        }

        // Expected results were evaluated using stan implementation
        expectedResult = std::vector<double>{3.87441, 4.94213, 3.52473, 6.43039, 4.06456};
        for (size_t i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            for (long j = statesMatrix.rows() / 2; j < statesMatrix.rows(); ++j) {
                chains[i].push_back(statesMatrix.row(j));
            }
        }

        actualResult = hops::computePotentialScaleReductionFactor(
                chains, numUnseen, sampleVariances, intraChainExpectations, interChainExpectation, numSeen);

        for (size_t i = 0; i < expectedResult.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult[i], actualResult[i], 0.01);
        }
    }

BOOST_AUTO_TEST_SUITE_END()