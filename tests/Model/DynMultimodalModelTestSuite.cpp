#define BOOST_TEST_MODULE DynMultimodalModelTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/Model/DynMultimodalModel.hpp>
#include "hops/Model/MultivariateGaussianModel.hpp"

BOOST_AUTO_TEST_SUITE(DynMultiModelTestSuite)
    template<typename Matrix, typename Vector>
    class ModelMock {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;

        typename MatrixType::Scalar fx;
        ModelMock (typename MatrixType::Scalar fx) : fx(fx) {
            // do nothing
        }

        double computeNegativeLogLikelihood(const VectorType &) const {
            return -std::log(fx);
        }

        [[nodiscard]] MatrixType computeExpectedFisherInformation(const VectorType &x) const {
            return Eigen::MatrixXd::Ones(x.rows(), x.rows());
        }

        [[nodiscard]] VectorType computeLogLikelihoodGradient(const VectorType &x) const {
            return x;
        }
    };

    BOOST_AUTO_TEST_CASE(CalculateNegativeLogLikelihood) {
        std::vector<double> componentLikelihoods({1, 2, 3, 4, 5});
        std::vector<double> weights({5, 4, 3, 2, 1});

        double expectedNegativeLogLikelihood = 0;
        std::vector<ModelMock<Eigen::MatrixXd, Eigen::VectorXd>> modelComponents;

        for (size_t i = 0; i < componentLikelihoods.size(); ++i) {
            modelComponents.push_back(ModelMock<Eigen::MatrixXd, Eigen::VectorXd>(componentLikelihoods[i]));
            expectedNegativeLogLikelihood += weights[i] * componentLikelihoods[i];
        }

        expectedNegativeLogLikelihood = -std::log(expectedNegativeLogLikelihood);

        auto multimodalModel = hops::DynMultimodalModel<ModelMock<Eigen::MatrixXd, Eigen::VectorXd>>(modelComponents, weights);
        auto evaluationPoint = Eigen::VectorXd::Zero(2);

        double actualNegativeLogLikelihood = multimodalModel.computeNegativeLogLikelihood(evaluationPoint);
        BOOST_CHECK_CLOSE(actualNegativeLogLikelihood, expectedNegativeLogLikelihood, 0.001);
    }

    BOOST_AUTO_TEST_CASE(CalculateNegativeLogLikelihoodForGaussianMixture) {
        unsigned d = 1;
        Eigen::VectorXd mean1 = Eigen::VectorXd::Ones(d);
        Eigen::VectorXd mean2 = -2 * Eigen::VectorXd::Ones(d);
        Eigen::MatrixXd covariance = 0.1 * Eigen::MatrixXd::Identity(d, d);

        hops::MultivariateGaussianModel model1(mean1, covariance);
        hops::MultivariateGaussianModel model2(mean2, covariance);

        hops::DynMultimodalModel multimodalModel = hops::DynMultimodalModel<hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>>({model1, model2}, {1, 1});

        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Zero(d);
        BOOST_CHECK_CLOSE(multimodalModel.computeNegativeLogLikelihood(evaluationPoint), 4.767645680805376, 0.001);

        evaluationPoint = 0.5 * Eigen::VectorXd::Ones(d);
        BOOST_CHECK_CLOSE(multimodalModel.computeNegativeLogLikelihood(evaluationPoint), 1.017645986707556, 0.001);
    }

    BOOST_AUTO_TEST_CASE(CalculateLogLikelihoodGradient) {
        unsigned d = 1;
        Eigen::VectorXd expectedLogLikelihoodGradient = 0.166667 * Eigen::VectorXd::Ones(d);

        hops::MultivariateGaussianModel model1 = hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>(Eigen::VectorXd::Zero(d), Eigen::MatrixXd::Identity(d, d));
        hops::MultivariateGaussianModel model2 = hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>(Eigen::VectorXd::Ones(d), Eigen::MatrixXd::Identity(d, d));

        Eigen::VectorXd evaluationPoint = 0.5 * Eigen::VectorXd::Ones(d);

        hops::DynMultimodalModel multimodalModel = hops::DynMultimodalModel<hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>>({model1, model2}, {1, 2});

        Eigen::VectorXd actualLogLikelihoodGradient = multimodalModel.computeLogLikelihoodGradient(evaluationPoint);

        BOOST_CHECK_SMALL((actualLogLikelihoodGradient - expectedLogLikelihoodGradient).squaredNorm(), 1.e-7);
    }
    
    BOOST_AUTO_TEST_CASE(CalculateLogLikelihoodGradientTrivial) {
        unsigned d = 1;
        Eigen::VectorXd expectedLogLikelihoodGradient = Eigen::VectorXd::Zero(d);

        hops::MultivariateGaussianModel model = hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>(Eigen::VectorXd::Zero(d), Eigen::MatrixXd::Identity(d, d));
        Eigen::VectorXd evaluationPoint = Eigen::VectorXd::Zero(d);

        hops::DynMultimodalModel multimodalModel = hops::DynMultimodalModel<hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>>({model, model}, {1, 2});

        Eigen::VectorXd actualLogLikelihoodGradient = multimodalModel.computeLogLikelihoodGradient(evaluationPoint);

        BOOST_CHECK_SMALL((actualLogLikelihoodGradient - expectedLogLikelihoodGradient).squaredNorm(), 1.e-7);
    }
BOOST_AUTO_TEST_SUITE_END()
