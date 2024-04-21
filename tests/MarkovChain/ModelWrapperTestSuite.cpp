#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ModelWrapperTestSuite

#include <boost/test/unit_test.hpp>
#include <memory>
#include <utility>

#include "hops/MarkovChain/ModelWrapper.hpp"
#include "hops/Model/Model.hpp"

namespace {
    class ModelMock : public hops::Model {
    private:
        double internal_state = 0;

    public:
        [[nodiscard]] hops::MatrixType::Scalar
        computeNegativeLogLikelihood(const Eigen::VectorXd &state) override {
            return state(0) + internal_state;
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<ModelMock>(*this);
        }

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override {
            return {"dummy variable name"};
        }
    };
}// namespace

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
