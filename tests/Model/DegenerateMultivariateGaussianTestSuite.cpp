#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DegenerateMultivariateGaussianModelTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>

#include "hops/Model/DegenerateGaussian.hpp"

BOOST_AUTO_TEST_SUITE(DegenerateMultivariateGaussianModel)

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihood) {
        double expectedValue = 1.98943678865;
        Eigen::VectorXd mean1(3);
        mean1 << -.8, -.8, .9;
        Eigen::VectorXd mean2(3);
        mean2 << .8, .8, .9;
        Eigen::MatrixXd covariance(3, 3);
        covariance << 0.04, 0, 0, 0, 0.04, 0, 0, 0, 500;

        std::vector<long> inactive_indices = {2};
        hops::DegenerateGaussian model1(mean1, covariance, inactive_indices);
        hops::DegenerateGaussian model2(mean2, covariance, inactive_indices);

        Eigen::VectorXd evaluationPoint(3);
        evaluationPoint << 0.8, 0.8, -2354;

        double actualValue =
                0.5 * (std::exp(-model1.computeNegativeLogLikelihood(evaluationPoint)) +
                       std::exp(-model2.computeNegativeLogLikelihood(evaluationPoint)));

        BOOST_CHECK_CLOSE(actualValue, expectedValue, 1e-2);
    }

    BOOST_AUTO_TEST_CASE(computeLogLikelihoodGradient) {
        Eigen::VectorXd mean(3);
        mean << -.8, -.8, 123;
        Eigen::MatrixXd covariance(3, 3);
        covariance << 0.04, 0, 0, 0, 0.04, 0, 0, 0, 500;

        std::vector<long> inactive_indices = {2};
        hops::DegenerateGaussian model(mean, covariance, inactive_indices);

        Eigen::VectorXd evaluationPoint1(3);
        evaluationPoint1 << 0.8, 0.8, 123;

        auto gradient = model.computeLogLikelihoodGradient(evaluationPoint1);
        if (gradient) {
            Eigen::VectorXd evaluationPoint2(3);
            evaluationPoint2 << evaluationPoint1(0) + 1e-5 * gradient.value()(0), evaluationPoint1(1) +
                                                                                  1e-5 * gradient.value()(1),
                    evaluationPoint1(2);

            // Tests if negative log likelihood decreases in gradient direction, thus checking correct sign of gradient.
            BOOST_CHECK_GT(model.computeNegativeLogLikelihood(evaluationPoint1),
                           model.computeNegativeLogLikelihood(evaluationPoint2));
        } else {
            BOOST_FAIL("Gradient was not available.");
        }
    }

    BOOST_AUTO_TEST_CASE(computeFisherInformation) {
        Eigen::VectorXd mean(2);
        mean << -.8, -.8;
        Eigen::MatrixXd covariance(2, 2);
        covariance << 8, 2, 2, 4;
        Eigen::MatrixXd expectedExpectedFisherInformation(2, 2);
        expectedExpectedFisherInformation << 1. / 7, -1. / 14, -1. / 14, 2. / 7;

        hops::DegenerateGaussian gaussian(mean, covariance);

        Eigen::VectorXd evaluationPoint1(2);
        evaluationPoint1 << 0.8, 0.8;

        auto actualExpectedFisherInformation1 = gaussian.computeExpectedFisherInformation(evaluationPoint1);
        if (actualExpectedFisherInformation1) {
            BOOST_CHECK(actualExpectedFisherInformation1.value().isApprox(expectedExpectedFisherInformation));
        } else {
            BOOST_FAIL("Fisher Info was not available.");
        }

        Eigen::VectorXd evaluationPoint2 = 5 * evaluationPoint1;
        auto actualExpectedFisherInformation2 = gaussian.computeExpectedFisherInformation(evaluationPoint2);
        if (actualExpectedFisherInformation2) {
            BOOST_CHECK(actualExpectedFisherInformation2.value().isApprox(expectedExpectedFisherInformation));
        } else {
            BOOST_FAIL("Fisher Info was not available.");
        }
    }

    BOOST_AUTO_TEST_CASE(computeFisherInformationWithInactiveIndices) {
        Eigen::VectorXd mean(3);
        mean << -.8, -.8, 123;
        Eigen::MatrixXd covariance(3, 3);
        covariance << 8, 2, 5, 2, 4, 5, 5, 5, 500;

        std::vector<long> inactive_indices = {2};
        hops::DegenerateGaussian model(mean, covariance, inactive_indices);

        Eigen::VectorXd evaluationPoint(3);
        evaluationPoint << 0.8, 0.8, 123;
        BOOST_CHECK(!model.computeExpectedFisherInformation(evaluationPoint));
    }

BOOST_AUTO_TEST_SUITE_END()
