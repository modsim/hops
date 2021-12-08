#define BOOST_TEST_MODULE ModelWrapperTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <memory>

#include <hops/Model/Model.hpp>
#include <hops/MarkovChain/ModelWrapper.hpp>
#include <utility>

namespace {
    class ModelMock : public hops::Model {
    public:
#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"
        [[nodiscard]] hops::MatrixType::Scalar
        computeNegativeLogLikelihood(const Eigen::VectorXd &state) const override {
            return state(0);
        }
#pragma clang diagnostic pop

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<ModelMock>();
        }
    };
}

BOOST_AUTO_TEST_SUITE(ModelWrapper)

    BOOST_AUTO_TEST_CASE(getModel) {
        std::shared_ptr<hops::Model> modelImp = std::make_shared<ModelMock>();
        auto model = hops::ModelWrapper(modelImp);

        BOOST_CHECK(model.getModel() == modelImp);
        BOOST_CHECK(model.hasModel() == true);
    }

    BOOST_AUTO_TEST_CASE(setModel) {
        auto modelImp = std::make_shared<ModelMock>();
        auto model = hops::ModelWrapper(modelImp);
        model.setModel(nullptr);
        BOOST_CHECK(model.getModel() == nullptr);
        BOOST_CHECK(model.hasModel() == false);
    }

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihood) {
        auto modelImp = std::make_shared<ModelMock>();
        auto model = hops::ModelWrapper(modelImp);
        BOOST_CHECK(model.computeNegativeLogLikelihood(hops::VectorType::Ones(1)) == 1);
    }

    BOOST_AUTO_TEST_CASE(computeLogLikelihoodGradient) {
        auto modelImp = std::make_shared<ModelMock>();
        auto model = hops::ModelWrapper(modelImp);
        BOOST_CHECK(model.computeLogLikelihoodGradient(hops::VectorType::Ones(1)) == std::nullopt);
    }

    BOOST_AUTO_TEST_CASE(computeExpectedFisherInformation) {
        auto modelImp = std::make_shared<ModelMock>();
        auto model = hops::ModelWrapper(modelImp);
        BOOST_CHECK(model.computeExpectedFisherInformation(hops::VectorType::Ones(1)) == std::nullopt);
    }

BOOST_AUTO_TEST_SUITE_END()
