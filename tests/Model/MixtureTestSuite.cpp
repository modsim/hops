#define BOOST_TEST_MODULE MixtureTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>

#include <hops/Model/Model.hpp>
#include <hops/Model/Mixture.hpp>
#include <hops/Model/MultivariateGaussian.hpp>

namespace {
    class ModelMock : public hops::Model {
        [[nodiscard]] double computeNegativeLogLikelihood(const hops::VectorType &) const override {
            return 3;
        }

        [[nodiscard]] std::optional<hops::VectorType>
        computeLogLikelihoodGradient(const hops::VectorType &x) const override {
            return x;
        }

        [[nodiscard]] std::optional<hops::MatrixType>
        computeExpectedFisherInformation(const hops::VectorType &x) const override {
            return Eigen::MatrixXd::Ones(x.rows(), x.rows());
        }
    };
}

BOOST_AUTO_TEST_SUITE(Mixture)

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihood) {
        double expectedNegativeLogLikelihood = 3;

        auto model1 = std::make_shared<ModelMock>();
        auto model2 = std::make_shared<ModelMock>();

        hops::Mixture mixture(std::vector<std::shared_ptr<hops::Model>>{model1, model2});

        auto evaluationPoint = Eigen::VectorXd::Zero(2);

        double actualNegativeLogLikelihood = mixture.computeNegativeLogLikelihood(evaluationPoint);
        BOOST_CHECK_EQUAL(actualNegativeLogLikelihood, expectedNegativeLogLikelihood);
    }

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihoodForGaussianMixture) {
        Eigen::VectorXd mean1 = Eigen::VectorXd::Ones(1);
        Eigen::VectorXd mean2 = -2 * Eigen::VectorXd::Ones(1);
        Eigen::MatrixXd covariance = 0.1 * Eigen::MatrixXd::Identity(1, 1);

        auto model1 = std::make_shared<hops::MultivariateGaussian>(mean1, covariance);
        auto model2 = std::make_shared<hops::MultivariateGaussian>(mean2, covariance);

        hops::Mixture mixture(std::vector<std::shared_ptr<hops::Model>>{model1, model2});


        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Zero(1);
        BOOST_CHECK_CLOSE(mixture.computeNegativeLogLikelihood(evaluationPoint), 5.46079, 0.001);

        evaluationPoint = 0.5 * Eigen::VectorXd::Ones(1);
        BOOST_CHECK_CLOSE(mixture.computeNegativeLogLikelihood(evaluationPoint), 1.71079, 0.001);

    }

    BOOST_AUTO_TEST_CASE(computeLogLikelihoodGradient) {
        Eigen::VectorXd expectedLogLikelihoodGradient = Eigen::VectorXd::Ones(2);
        auto model1 = std::make_shared<ModelMock>();
        auto model2 = std::make_shared<ModelMock>();

        hops::Mixture mixture(std::vector<std::shared_ptr<hops::Model>>{model1, model2});

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(2);

        Eigen::VectorXd actualLogLikelihoodGradient = mixture.computeLogLikelihoodGradient(evaluationPoint).value();
        BOOST_CHECK(actualLogLikelihoodGradient == expectedLogLikelihoodGradient);
    }

    BOOST_AUTO_TEST_CASE(CalculateLogLikelihoodGradientForGaussianMixture) {
        unsigned d = 1;
        Eigen::VectorXd expectedLogLikelihoodGradient = 0.166667 * Eigen::VectorXd::Ones(d);

        auto model1 = std::make_shared<hops::MultivariateGaussian>(Eigen::VectorXd::Zero(d),
                                                                        Eigen::MatrixXd::Identity(d, d));
        auto model2 = std::make_shared<hops::MultivariateGaussian>(Eigen::VectorXd::Ones(d),
                                                                        Eigen::MatrixXd::Identity(d, d));

        hops::Mixture mixture(std::vector<std::shared_ptr<hops::Model>>{model1, model2}, {1, 2});

        Eigen::VectorXd evaluationPoint = 0.5 * Eigen::VectorXd::Ones(d);
        auto actualLogLikelihoodGradient = mixture.computeLogLikelihoodGradient(evaluationPoint);
        if(actualLogLikelihoodGradient) {
            BOOST_CHECK_SMALL((actualLogLikelihoodGradient.value() - expectedLogLikelihoodGradient).squaredNorm(), 1.e-7);
        }
        else {
            BOOST_FAIL("Expected Gradient to be present");
        }
    }


    BOOST_AUTO_TEST_CASE(checkExpectedFisherInformationReturnsEmpty) {
        auto model1 = std::make_shared<ModelMock>();
        auto model2 = std::make_shared<ModelMock>();

        hops::Mixture mixture(std::vector<std::shared_ptr<hops::Model>>{model1, model2});

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(2);

        auto actualExpectedFisherInformation = mixture.computeExpectedFisherInformation(evaluationPoint);
        BOOST_CHECK(!actualExpectedFisherInformation.has_value());
    }

BOOST_AUTO_TEST_SUITE_END()