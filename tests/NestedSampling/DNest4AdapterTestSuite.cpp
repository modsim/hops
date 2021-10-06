#define BOOST_TEST_MODULE DNest4AdapterTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/NestedSampling/DNest4Adapter.hpp>
#include <Eigen/Core>
#include <hops/Model/MultivariateGaussianModel.hpp>

BOOST_AUTO_TEST_SUITE(DNest4AdapterTestSuite)
BOOST_AUTO_TEST_CASE(TestGaussianModelEvidenceIsCorrect) {
    Eigen::MatrixXd A(2, 1);
    A << 1, -1;
    Eigen::VectorXd b(2);
    b << 5, 5;

    Eigen::MatrixXd cov(1, 1);
    cov << 1;
    Eigen::VectorXd mu(1);
    mu << 0;

    auto model = hops::MultivariateGaussianModel(mu, cov);

    auto priorSampler = ;

    auto posteriorSampler;

}
BOOST_AUTO_TEST_SUITE_END()
