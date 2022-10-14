#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ExpectecSquaredJumpDistanceTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "hops/FileReader/CsvReader.hpp"
#include "hops/Statistics/Covariance.hpp"

BOOST_AUTO_TEST_SUITE(CovarianceTestSuite)
    BOOST_AUTO_TEST_CASE(ComputeFull) {
        auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/test_states/states0.csv");
        Eigen::MatrixXd expectedCovariance(5, 5);
        expectedCovariance << 
                0.03232092, -0.08417842,  0.05216992, -0.12575977, -0.04064512,
                -0.08417842,  0.36535426, -0.24331461,  0.57510415,  0.1586517 ,
                 0.05216992, -0.24331461,  0.19777048, -0.44320767, -0.1336759 ,
                -0.12575977,  0.57510415, -0.44320767,  1.03112642,  0.3256675 ,
                -0.04064512,  0.1586517 , -0.1336759 ,  0.3256675 ,  0.15002477;

        std::vector<Eigen::VectorXd> draws(statesMatrix.rows());

        for (long j = 0; j < statesMatrix.rows(); ++j) {
            draws[j] = statesMatrix.row(j);
        }

        std::vector<std::vector<Eigen::VectorXd>> chains = {draws};
        auto covariance = hops::computeCovariance<Eigen::VectorXd, Eigen::MatrixXd>(chains);
        
        for (long i = 0; i < 5*5; ++i) {
            BOOST_CHECK_CLOSE(expectedCovariance(i), covariance(i), 1.e-5);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeIncrementally) {
        auto statesMatrix = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/test_states/states0.csv");
        Eigen::MatrixXd expectedCovarianceHalf(5, 5);
        expectedCovarianceHalf << 
                 0.0194571 , -0.04005641,  0.02119723, -0.05728292, -0.0396404 ,
                -0.04005641,  0.13692184, -0.07798504,  0.23698141,  0.15482367,
                 0.02119723, -0.07798504,  0.04540932, -0.13731044, -0.08869793,
                -0.05728292,  0.23698141, -0.13731044,  0.43917515,  0.28136163,
                -0.0396404 ,  0.15482367, -0.08869793,  0.28136163,  0.18226357;

        Eigen::MatrixXd expectedCovarianceFull(5, 5);
        expectedCovarianceFull << 
                 0.03232092, -0.08417842,  0.05216992, -0.12575977, -0.04064512,
                -0.08417842,  0.36535426, -0.24331461,  0.57510415,  0.1586517 ,
                 0.05216992, -0.24331461,  0.19777048, -0.44320767, -0.1336759 ,
                -0.12575977,  0.57510415, -0.44320767,  1.03112642,  0.3256675 ,
                -0.04064512,  0.1586517 , -0.1336759 ,  0.3256675 ,  0.15002477;

        long firstHalf = (statesMatrix.rows() / 2);

        std::vector<Eigen::VectorXd> draws(firstHalf);

        for (long i = 0; i < firstHalf; ++i) {
            draws[i] = statesMatrix.row(i);
        }

        std::vector<std::vector<Eigen::VectorXd>> chains = {draws};
        auto[covariance, intermediateResult] = hops::computeCovarianceIncrementally<Eigen::VectorXd, Eigen::MatrixXd>(chains);
        
        for (long i = 0; i < 5*5; ++i) {
            BOOST_CHECK_CLOSE(expectedCovarianceHalf(i), covariance(i), 1.e-5);
        }

        chains[0].resize(statesMatrix.rows());
        for (long i = firstHalf; i < statesMatrix.rows(); ++i) {
            chains[0][i] = statesMatrix.row(i);
        }

        std::tie(covariance, intermediateResult) = hops::computeCovarianceIncrementally<Eigen::VectorXd, Eigen::MatrixXd>(chains, intermediateResult);
        
        for (long i = 0; i < 5*5; ++i) {
            BOOST_CHECK_CLOSE(expectedCovarianceFull(i), covariance(i), 1.e-5);
        }
    }
BOOST_AUTO_TEST_SUITE_END()
