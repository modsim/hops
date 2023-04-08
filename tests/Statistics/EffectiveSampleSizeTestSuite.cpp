#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE EffectiveSampleSizeTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#include "hops/FileReader/CsvReader.hpp"
#include "hops/Statistics/EffectiveSampleSize.hpp"

BOOST_AUTO_TEST_SUITE(EffectiveSampleSizeTestSuite)
    BOOST_AUTO_TEST_CASE(ComputeOneDimSingleChain) {
        // evaluated using stan implementation
        std::vector<Eigen::VectorXd> draws = {
            0*Eigen::VectorXd::Ones(1), 
            1*Eigen::VectorXd::Ones(1), 
            3*Eigen::VectorXd::Ones(1), 
            2*Eigen::VectorXd::Ones(1), 
            4*Eigen::VectorXd::Ones(1), 
            2*Eigen::VectorXd::Ones(1),
            1*Eigen::VectorXd::Ones(1),
            6*Eigen::VectorXd::Ones(1)
        };
        std::vector<std::vector<Eigen::VectorXd>> chains = {draws};

        double expectedResult = 7.22472;
        double actualResult = hops::computeEffectiveSampleSize(chains, 0);

        BOOST_CHECK_CLOSE(expectedResult, actualResult, 0.01);
    }

    BOOST_AUTO_TEST_CASE(ComputeMultiDimSingleChain) {
        auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/test_states/states0.csv");

        std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

        for (long j = 0; j < statesMatrix.rows(); ++j) {
            draws[j] = statesMatrix.row(j);
        }

        std::vector<std::vector<Eigen::VectorXd>> chains = {draws};

        // evaluated using stan implementation
        std::vector<double> expectedResult = {9.66671, 3.70636, 3.06245, 3.13602, 5.25293};
        std::vector<double> actualResult = hops::computeEffectiveSampleSize(chains);

        for (long d = 0; d < draws[0].size(); ++d) {
            BOOST_CHECK_CLOSE(expectedResult[d], actualResult[d], 0.01);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeMultiDimMultiChain) {
        // evaluated using stan implementation
        std::vector<std::vector<Eigen::VectorXd>> chains;

        for (unsigned i = 0; i < 4; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

            for (long j = 0; j < statesMatrix.rows(); ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains.push_back(draws);
        }

        std::vector<double> expectedResult = {2.5219, 2.45149, 2.5539, 2.4103, 2.50195};
        std::vector<double> actualResult = hops::computeEffectiveSampleSize(chains);

        for (long d = 0; d < chains[0][0].size(); ++d) {
            BOOST_CHECK_CLOSE(expectedResult[d], actualResult[d], 20);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeMultiDimMultiChainPointerArray) {
        unsigned numChains = 4;
        std::vector<std::vector<Eigen::VectorXd>> chains(numChains);
        std::vector<const std::vector<Eigen::VectorXd>*> chainsPtrArray(numChains);

        for (unsigned i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/states") + std::to_string(i) + ".csv");

            std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

            for (long j = 0; j < statesMatrix.rows(); ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains[i] = draws;
            chainsPtrArray[i] = &chains[i];
        }

        std::vector<double> expectedResult = {2.5219, 2.45149, 2.5539, 2.4103, 2.50195};
        std::vector<double> actualResult = hops::computeEffectiveSampleSize(chainsPtrArray);

        for (long d = 0; d < (*chainsPtrArray[0])[0].size(); ++d) {
            BOOST_CHECK_CLOSE(expectedResult[d], actualResult[d], 20);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeMultiDimMultiChainLarge) {
        // evaluated using stan implementation
        std::vector<std::vector<Eigen::VectorXd>> chains;

        for (unsigned i = 0; i < 4; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/chain") + std::to_string(i) + "_states.csv");

            std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

            for (long j = 0; j < statesMatrix.rows(); ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains.push_back(draws);
        }

        std::vector<double> expectedResult = {10259.3, 10600.8, 10670.2, 10097.2, 10382.7};
        std::vector<double> actualResult = hops::computeEffectiveSampleSize(chains);

        for (long d = 0; d < chains[0][0].size(); ++d) {
            BOOST_CHECK_CLOSE(expectedResult[d], actualResult[d], 1);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeMultiDimMultiChainLargePointerArray) {
        unsigned numChains = 4;
        std::vector<std::vector<Eigen::VectorXd>> chains(numChains);
        std::vector<const std::vector<Eigen::VectorXd>*> chainsPtrArray(numChains);

        for (unsigned i = 0; i < numChains; ++i) {
            auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                    std::string("../../resources/test_states/chain") + std::to_string(i) + "_states.csv");

            std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

            for (long j = 0; j < statesMatrix.rows(); ++j) {
                draws[j] = statesMatrix.row(j);
            }

            chains[i] = draws;
            chainsPtrArray[i] = &chains[i];
        }

        std::vector<double> expectedResult = {10259.3, 10600.8, 10670.2, 10097.2, 10382.7};
        std::vector<double> actualResult = hops::computeEffectiveSampleSize(chainsPtrArray);

        for (long d = 0; d < (*chainsPtrArray[0])[0].size(); ++d) {
            BOOST_CHECK_CLOSE(expectedResult[d], actualResult[d], 1);
        }
    }
BOOST_AUTO_TEST_SUITE_END()
