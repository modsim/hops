#define BOOST_TEST_MODULE MultimodalModelTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/Model/MultimodalModel.hpp>

namespace {
    template<typename Matrix, typename Vector>
    class ModelMock {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;

        static double calculateNegativeLogLikelihood(const VectorType &) {
            return 21;
        }

        [[nodiscard]] MatrixType calculateExpectedFisherInformation(const VectorType &x) const {
            return Eigen::MatrixXd::Ones(x.rows(), x.rows());
        }

        [[nodiscard]] VectorType calculateLogLikelihoodGradient(const VectorType &x) const {
            return x;
        }
    };
}

BOOST_AUTO_TEST_SUITE(MultimodalModel)

    BOOST_AUTO_TEST_CASE( calculateNegativeLogLikelihood) {
        double expectedNegativeLogLikelihood = 21;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model1;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model2;

        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        auto evaluationPoint = Eigen::VectorXd::Zero(2);

        double actualNegativeLogLikelihood = multimodalModel.calculateNegativeLogLikelihood(evaluationPoint);
        BOOST_CHECK(actualNegativeLogLikelihood == expectedNegativeLogLikelihood);
    }

    BOOST_AUTO_TEST_CASE( calculateNegativeLogLikelihoodForGaussianMixture) {
        Eigen::VectorXd mean1 = Eigen::VectorXd::Ones(1);
        Eigen::VectorXd mean2 = -Eigen::VectorXd::Ones(1);
        Eigen::MatrixXd covariance = 0.1 * Eigen::MatrixXd::Identity(1, 1);

        hops::MultivariateGaussianModel model1(mean1, covariance);
        hops::MultivariateGaussianModel model2(mean2, covariance);
        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Zero(1);
        BOOST_CHECK_CLOSE(multimodalModel.calculateNegativeLogLikelihood(evaluationPoint), 4.767646, 0.001);

        evaluationPoint = 0.75*Eigen::VectorXd::Ones(1);
        BOOST_CHECK_CLOSE(multimodalModel.calculateNegativeLogLikelihood(evaluationPoint), 0.773292, 0.001);
    }

    BOOST_AUTO_TEST_CASE( calculateExpectedFisherInformation) {
        Eigen::MatrixXd expectedExpectedFisherInformation = 2 * Eigen::MatrixXd::Ones(2, 2);
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model1;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model2;

        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(2);

        Eigen::MatrixXd actualExpectedFisherInformation = multimodalModel.calculateExpectedFisherInformation(
                evaluationPoint);
        BOOST_CHECK(actualExpectedFisherInformation == expectedExpectedFisherInformation);
    }

    BOOST_AUTO_TEST_CASE( calculateLogLikelihoodGradient) {
        Eigen::VectorXd expectedLogLikelihoodGradient = 2 * Eigen::VectorXd::Ones(2);
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model1;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model2;

        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(2);

        Eigen::VectorXd actualLogLikelihoodGradient = multimodalModel.calculateLogLikelihoodGradient(evaluationPoint);
        BOOST_CHECK(actualLogLikelihoodGradient == expectedLogLikelihoodGradient);
    }

    BOOST_AUTO_TEST_SUITE_END()