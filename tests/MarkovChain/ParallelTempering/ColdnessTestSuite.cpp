#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ColdnessTestSuite

#include <boost/test/unit_test.hpp>

#include "hops/MarkovChain/ParallelTempering/Coldness.hpp"
#include "hops/Utility/VectorType.hpp"
#include "hops/Model/Model.hpp"

namespace {
    class ModelMock : public hops::Model {
    public:
        ModelMock() = default;

        explicit ModelMock(double negativeLogLikelihoodConstant) :
                negativeLogLikelihoodConstant(negativeLogLikelihoodConstant) {}

        [[nodiscard]] double computeNegativeLogLikelihood(const hops::VectorType &) override {
            return negativeLogLikelihoodConstant;
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<ModelMock>();
        }

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override {
            return {"dummy variable name"};
        }

    private:
        double negativeLogLikelihoodConstant = 0;
    };

    class ModelMockWithGradient : public hops::Model {
    public:
        ModelMockWithGradient() = default;

        explicit ModelMockWithGradient(double negativeLogLikelihoodConstant) :
                negativeLogLikelihoodConstant(negativeLogLikelihoodConstant) {}

        [[nodiscard]] double computeNegativeLogLikelihood(const hops::VectorType &) override {
            return negativeLogLikelihoodConstant;
        }

        [[nodiscard]] std::optional<hops::VectorType>
        computeLogLikelihoodGradient(const hops::VectorType &) override {
            return hops::VectorType::Ones(1);
        }

        [[nodiscard]] std::optional<hops::MatrixType>
        computeExpectedFisherInformation(const hops::VectorType &) override {
            return hops::MatrixType::Identity(1, 1);
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<ModelMockWithGradient>(*this);
        }


        [[nodiscard]] std::vector<std::string> getDimensionNames() const override {
            return {"dummy variable name"};
        }

    private:
        double negativeLogLikelihoodConstant = 0;
    };
}

BOOST_AUTO_TEST_SUITE(Coldness)

    BOOST_AUTO_TEST_CASE(getColdness) {
        double expectedColdness = 0.5;
        auto modelMock = ModelMock();
        hops::Coldness modelWithColdness(modelMock, expectedColdness);

        double actualColdness = modelWithColdness.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE(setColdness) {
        double expectedColdness = 0.5;

        auto modelMock = ModelMock();
        hops::Coldness modelWithColdness(modelMock);

        modelWithColdness.setColdness(expectedColdness);

        double actualColdness = modelWithColdness.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE(constructorWithColdnessLargerThan1) {
        double expectedColdness = 1;

        auto modelMock = ModelMock();
        hops::Coldness modelWithColdness(modelMock, 1.5);

        double actualColdness = modelWithColdness.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE(constructorWithColdnessLessThan0) {
        double expectedColdness = 0;

        auto modelMock = ModelMock();
        hops::Coldness markovChainWithColdness(modelMock, -1.5);

        double actualColdness = markovChainWithColdness.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE(setColdnessToMoreThan1G) {
        double expectedColdness = 1;

        auto modelMock = ModelMock();
        hops::Coldness markovChainWithColdness(modelMock, 1.5);

        markovChainWithColdness.setColdness(1.5);

        double actualColdness = markovChainWithColdness.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE(setColdnessToLessThan0) {
        double expectedColdness = 0;

        auto modelMock = ModelMock();
        hops::Coldness markovChainWithColdness(modelMock, -1.5);

        markovChainWithColdness.setColdness(-1.5);

        double actualColdness = markovChainWithColdness.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE(computeNegativeLogLikelihood) {
        double expectedNegativeLogLikelihood = -250;

        auto modelMock = ModelMock(-1000);
        hops::Coldness markovChainWithColdness(modelMock, 0.25);
        hops::VectorType mockState = Eigen::VectorXd::Zero(1);
        double actualNegativeLogLikelihood = markovChainWithColdness.computeNegativeLogLikelihood(mockState);
        BOOST_CHECK(actualNegativeLogLikelihood == expectedNegativeLogLikelihood);
    }

    BOOST_AUTO_TEST_CASE(computeNulloptGradient) {
        auto modelMock = ModelMock();
        hops::Coldness markovChainWithColdness(modelMock);
        hops::VectorType mockState = Eigen::VectorXd::Zero(1);
        auto gradient = markovChainWithColdness.computeLogLikelihoodGradient(mockState);
        BOOST_CHECK(!gradient.has_value());
    }

    BOOST_AUTO_TEST_CASE(computeNulloptFisherInformation) {
        auto modelMock = ModelMock();
        hops::Coldness markovChainWithColdness(modelMock);
        hops::VectorType mockState = Eigen::VectorXd::Zero(1);
        auto fisherInformation = markovChainWithColdness.computeExpectedFisherInformation(mockState);
        BOOST_CHECK(!fisherInformation.has_value());
    }

    BOOST_AUTO_TEST_CASE(computeGradient) {
        auto modelMock = ModelMockWithGradient();
        double coldness = 0.5;
        hops::Coldness markovChainWithColdness(modelMock, coldness);
        hops::VectorType mockState = Eigen::VectorXd::Zero(1);
        auto gradient = markovChainWithColdness.computeLogLikelihoodGradient(mockState);
        BOOST_REQUIRE(gradient.has_value());
        if (gradient) {
            BOOST_CHECK(gradient = mockState * coldness);
        }
    }

    BOOST_AUTO_TEST_CASE(computeFisherInformation) {
        auto modelMock = ModelMockWithGradient();
        double coldness = 0.5;
        hops::Coldness markovChainWithColdness(modelMock, coldness);
        hops::VectorType mockState = Eigen::VectorXd::Zero(1);
        auto fisherInformation = markovChainWithColdness.computeExpectedFisherInformation(mockState);
        BOOST_REQUIRE(fisherInformation.has_value());
        if (fisherInformation) {
            // coldness enters twice because fisherInformation is jacobian^T * cov^-1 * jacobian
            BOOST_CHECK(fisherInformation = coldness * mockState * coldness);
        }
    }

BOOST_AUTO_TEST_SUITE_END()
