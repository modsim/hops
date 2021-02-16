#define BOOST_TEST_MODULE MultivariateGaussianModelTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/Model/MultivariateGaussianModel.hpp>

BOOST_AUTO_TEST_SUITE(MultivariateGaussianModel)

    BOOST_AUTO_TEST_CASE(calculateNegativeLogLikelihood) {
        double expectedValue = 1.98943678865;
        Eigen::VectorXd mean1(2);
        mean1 << -.8, -.8;
        Eigen::VectorXd mean2(2);
        mean2 << .8, .8;
        Eigen::MatrixXd covariance(2, 2);
        covariance << 0.04, 0, 0, 0.04;

        hops::MultivariateGaussianModel multivariateGaussianModel1(mean1, covariance);
        hops::MultivariateGaussianModel multivariateGaussianModel2(mean2, covariance);

        Eigen::VectorXd evaluationPoint(2);
        evaluationPoint << 0.8, 0.8;

        double actualValue =
                0.5 * (std::exp(-multivariateGaussianModel1.calculateNegativeLogLikelihood(evaluationPoint)) +
                       std::exp(-multivariateGaussianModel2.calculateNegativeLogLikelihood(evaluationPoint)));

        BOOST_CHECK_CLOSE(actualValue, expectedValue, 1e-2);
    }

    BOOST_AUTO_TEST_CASE(calculateLogLikelihoodGradient) {
        Eigen::VectorXd mean(2);
        mean << -.8, -.8;
        Eigen::MatrixXd covariance(2, 2);
        covariance << 0.04, 0, 0, 0.04;

        hops::MultivariateGaussianModel multivariateGaussianModel(mean, covariance);

        Eigen::VectorXd evaluationPoint1(2);
        evaluationPoint1 << 0.8, 0.8;

        Eigen::VectorXd evaluationPoint2 =
                evaluationPoint1 + 1e-5 * multivariateGaussianModel.calculateLogLikelihoodGradient(evaluationPoint1);

        // Tests if negative log likelihood decreases in gradient direction, thus checking correct sign of gradient.
        BOOST_CHECK_GT(multivariateGaussianModel.calculateNegativeLogLikelihood(evaluationPoint1),
                  multivariateGaussianModel.calculateNegativeLogLikelihood(evaluationPoint2));
    }

BOOST_AUTO_TEST_SUITE_END()