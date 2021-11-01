#define BOOST_TEST_MODULE MultivariateGaussianModelTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/Model/MultivariateGaussian.hpp>

BOOST_AUTO_TEST_SUITE(MultivariateGaussianModel)

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihood) {
        double expectedValue = 1.98943678865;
        Eigen::VectorXd mean1(2);
        mean1 << -.8, -.8;
        Eigen::VectorXd mean2(2);
        mean2 << .8, .8;
        Eigen::MatrixXd covariance(2, 2);
        covariance << 0.04, 0, 0, 0.04;

        hops::MultivariateGaussian multivariateGaussianModel1(mean1, covariance);
        hops::MultivariateGaussian multivariateGaussianModel2(mean2, covariance);

        Eigen::VectorXd evaluationPoint(2);
        evaluationPoint << 0.8, 0.8;

        double actualValue =
                0.5 * (std::exp(-multivariateGaussianModel1.computeNegativeLogLikelihood(evaluationPoint)) +
                       std::exp(-multivariateGaussianModel2.computeNegativeLogLikelihood(evaluationPoint)));

        BOOST_CHECK_CLOSE(actualValue, expectedValue, 1e-2);
    }

    BOOST_AUTO_TEST_CASE(computeLogLikelihoodGradient) {
        Eigen::VectorXd mean(2);
        mean << -.8, -.8;
        Eigen::MatrixXd covariance(2, 2);
        covariance << 0.04, 0, 0, 0.04;

        hops::MultivariateGaussian multivariateGaussianModel(mean, covariance);

        Eigen::VectorXd evaluationPoint1(2);
        evaluationPoint1 << 0.8, 0.8;

        auto gradient = multivariateGaussianModel.computeLogLikelihoodGradient(evaluationPoint1);
        if (gradient) {
            Eigen::VectorXd evaluationPoint2 = evaluationPoint1 + 1e-5 * gradient.value();

            // Tests if negative log likelihood decreases in gradient direction, thus checking correct sign of gradient.
            BOOST_CHECK_GT(multivariateGaussianModel.computeNegativeLogLikelihood(evaluationPoint1),
                           multivariateGaussianModel.computeNegativeLogLikelihood(evaluationPoint2));
        }
        else {
            BOOST_FAIL("Gradient was not available.");
        }
    }

    BOOST_AUTO_TEST_CASE(computeFisherInformation) {
        Eigen::VectorXd mean(2);
        mean << -.8, -.8;
        Eigen::MatrixXd covariance(2, 2);
        covariance << 8, 2, 2, 4;
        Eigen::MatrixXd expectedExpectedFisherInformation(2, 2);
        expectedExpectedFisherInformation << 1./7, -1./14, -1./14, 2./7;

        hops::MultivariateGaussian multivariateGaussianModel(mean, covariance);

        Eigen::VectorXd evaluationPoint1(2);
        evaluationPoint1 << 0.8, 0.8;

        auto actualExpectedFisherInformation1 = multivariateGaussianModel.computeExpectedFisherInformation(evaluationPoint1);
        if (actualExpectedFisherInformation1) {
            BOOST_CHECK(actualExpectedFisherInformation1.value().isApprox(expectedExpectedFisherInformation));
        }
        else {
            BOOST_FAIL("Fisher Info was not available.");
        }

        Eigen::VectorXd evaluationPoint2 = 5*evaluationPoint1;
        auto actualExpectedFisherInformation2 = multivariateGaussianModel.computeExpectedFisherInformation(evaluationPoint2);
        if (actualExpectedFisherInformation2) {
            BOOST_CHECK(actualExpectedFisherInformation2.value().isApprox(expectedExpectedFisherInformation));
        }
        else {
            BOOST_FAIL("Fisher Info was not available.");
        }
    }

BOOST_AUTO_TEST_SUITE_END()