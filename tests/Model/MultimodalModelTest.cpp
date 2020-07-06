#include <Eigen/Core>
#include <gtest/gtest.h>
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

    TEST(MultimodalModel, calculateNegativeLogLikelihood) {
        double expectedNegativeLogLikelihood = 21;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model1;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model2;

        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        auto evaluationPoint = Eigen::VectorXd::Zero(2);

        double actualNegativeLogLikelihood = multimodalModel.calculateNegativeLogLikelihood(evaluationPoint);
        EXPECT_EQ(actualNegativeLogLikelihood, expectedNegativeLogLikelihood);
    }

    TEST(MultimodalModel, calculateNegativeLogLikelihoodForGaussianMixture) {
        Eigen::VectorXd mean1 = Eigen::VectorXd::Ones(1);
        Eigen::VectorXd mean2 = -Eigen::VectorXd::Ones(1);
        Eigen::MatrixXd covariance = 0.1 * Eigen::MatrixXd::Identity(1, 1);

        hops::MultivariateGaussianModel model1(mean1, covariance);
        hops::MultivariateGaussianModel model2(mean2, covariance);
        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Zero(1);
        EXPECT_NEAR(multimodalModel.calculateNegativeLogLikelihood(evaluationPoint), 4.767646, 0.001);

        evaluationPoint = 0.75*Eigen::VectorXd::Ones(1);
        EXPECT_NEAR(multimodalModel.calculateNegativeLogLikelihood(evaluationPoint), 0.773292, 0.001);
    }

    TEST(MultimodalModel, calculateExpectedFisherInformation) {
        Eigen::MatrixXd expectedExpectedFisherInformation = 2 * Eigen::MatrixXd::Ones(2, 2);
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model1;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model2;

        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(2);

        Eigen::MatrixXd actualExpectedFisherInformation = multimodalModel.calculateExpectedFisherInformation(
                evaluationPoint);
        EXPECT_EQ(actualExpectedFisherInformation, expectedExpectedFisherInformation);
    }

    TEST(MultimodalModel, calculateLogLikelihoodGradient) {
        Eigen::VectorXd expectedLogLikelihoodGradient = 2 * Eigen::VectorXd::Ones(2);
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model1;
        ModelMock<Eigen::MatrixXd, Eigen::VectorXd> model2;

        hops::MultimodalModel multimodalModel(std::make_tuple(model1, model2));

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Ones(2);

        Eigen::VectorXd actualLogLikelihoodGradient = multimodalModel.calculateLogLikelihoodGradient(evaluationPoint);
        EXPECT_EQ(actualLogLikelihoodGradient, expectedLogLikelihoodGradient);
    }
}
