#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MixtureTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>

#include "hops/Model/Model.hpp"
#include "hops/Model/Mixture.hpp"
#include "hops/Model/Gaussian.hpp"

namespace {
    class ModelMock : public hops::Model {
    public:
        [[nodiscard]] double computeNegativeLogLikelihood(const hops::VectorType &) override {
            return 3;
        }

        [[nodiscard]] std::optional<hops::VectorType>
        computeLogLikelihoodGradient(const hops::VectorType &x) override {
            return x;
        }

        [[nodiscard]] std::optional<hops::MatrixType>
        computeExpectedFisherInformation(const hops::VectorType &x) override {
            return Eigen::MatrixXd::Ones(x.rows(), x.rows());
        }

        std::vector<std::string> getDimensionNames() const override {
            return std::vector<std::string> {"dummy name"};
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<ModelMock>();
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

        auto model1 = std::make_shared<hops::Gaussian>(mean1, covariance);
        auto model2 = std::make_shared<hops::Gaussian>(mean2, covariance);

        hops::Mixture mixture(std::vector<std::shared_ptr<hops::Model>>{model1, model2});


        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Zero(1);
        BOOST_CHECK_CLOSE(mixture.computeNegativeLogLikelihood(evaluationPoint), 5.46079, 0.001);

        evaluationPoint = 0.5 * Eigen::VectorXd::Ones(1);
        BOOST_CHECK_CLOSE(mixture.computeNegativeLogLikelihood(evaluationPoint), 1.71079, 0.001);

    }

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihoodGradientForGaussianMixtureAtLowDensityPoint) {
        unsigned d = 2;

        Eigen::VectorXd mean(d);
        Eigen::MatrixXd covariance = 2 * Eigen::MatrixXd::Identity(d, d);
        mean << -2., 0.;
        auto model1 = std::make_shared<hops::Gaussian>(mean, covariance);
        mean << 2., 0.;
        covariance << 5., 0.,
                0., 0.2;
        auto model2 = std::make_shared<hops::Gaussian>(mean, covariance);

        hops::Mixture mixture(std::vector<std::shared_ptr<hops::Model>>{model1, model2}, {0.5, 0.5});

        Eigen::VectorXd evaluationPoint(d);
        evaluationPoint << 62., 100.;
        double actualNegativeLogLikelihood = mixture.computeNegativeLogLikelihood(evaluationPoint);
        BOOST_CHECK_CLOSE(actualNegativeLogLikelihood, 3527.22417, 0.001);
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

        auto model1 = std::make_shared<hops::Gaussian>(Eigen::VectorXd::Zero(d),
                                                       Eigen::MatrixXd::Identity(d, d));
        auto model2 = std::make_shared<hops::Gaussian>(Eigen::VectorXd::Ones(d),
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
