#define BOOST_TEST_MODULE GammaModelTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/Model/Gamma.hpp>

BOOST_AUTO_TEST_SUITE(GammaModel)

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihood2D) {
        double expectedValue = -2.53055791067;

        hops::Gamma gammaModel(Eigen::VectorXd::Ones(1));

        Eigen::VectorXd evaluationPoint(2);
        evaluationPoint << 2, 3;

        double actualValue = gammaModel.computeNegativeLogLikelihood(evaluationPoint);

        BOOST_CHECK_CLOSE(actualValue, expectedValue, 1e-10);
    }

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihood4D) {
        double expectedValue = 2 * -2.530557910669552716123823807178384;

        hops::Gamma gammaModel(Eigen::VectorXd::Ones(2));

        Eigen::VectorXd evaluationPoint(4);
        evaluationPoint << 2, 3, 2, 3;

        double actualValue = gammaModel.computeNegativeLogLikelihood(evaluationPoint);

        BOOST_CHECK_CLOSE(actualValue, expectedValue, 1e-10);
    }

    BOOST_AUTO_TEST_CASE(computeGradient2D) {
        Eigen::VectorXd expectedValue(2);
        expectedValue << -std::log(3) - boost::math::digamma(2), -5. / 9;


        hops::Gamma gammaModel(Eigen::VectorXd::Ones(1));

        Eigen::VectorXd evaluationPoint(2);
        evaluationPoint << 2, 3;

        Eigen::VectorXd actualValue = gammaModel.computeLogLikelihoodGradient(evaluationPoint).value();

        BOOST_CHECK(actualValue == expectedValue);
    }

    BOOST_AUTO_TEST_CASE(computeFisherInformation2D) {
        Eigen::MatrixXd expectedValue(2, 2);
        expectedValue << boost::math::polygamma(1, 2.), 1. / 3, 1. / 3, 2. / std::pow(3,2);


        hops::Gamma gammaModel(Eigen::VectorXd::Ones(1));

        Eigen::VectorXd evaluationPoint(2);
        evaluationPoint << 2, 3;

        Eigen::MatrixXd actualValue = gammaModel.computeExpectedFisherInformation(evaluationPoint).value();

        BOOST_CHECK(actualValue == expectedValue);
    }

BOOST_AUTO_TEST_SUITE_END()

