#define BOOST_TEST_MODULE ColdnessAttributeTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/MarkovChain/ParallelTempering/ColdnessAttribute.hpp>

BOOST_AUTO_TEST_SUITE(ColdnessAttribute)

BOOST_AUTO_TEST_CASE( getColdness) {
        double expectedColdness = 0.5;

        class ModelMock {
        public:
            using VectorType = Eigen::VectorXd;
            using MatrixType = Eigen::MatrixXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, expectedColdness);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE( setColdness) {
        double expectedColdness = 0.5;

        class ModelMock {
        public:
            using VectorType = Eigen::VectorXd;
            using MatrixType = Eigen::MatrixXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 1.);

        markovChainWithColdnessAttribute.setColdness(expectedColdness);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE( constructorWithColdnessLargerThan1) {
        double expectedColdness = 1;

        class ModelMock {
        public:
            using VectorType = Eigen::VectorXd;
            using MatrixType = Eigen::MatrixXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE( constructorWithColdnessLessThan0) {
        double expectedColdness = 0;

        class ModelMock {
        public:
            using VectorType = Eigen::VectorXd;
            using MatrixType = Eigen::MatrixXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, -1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE( setColdnessToMoreThan1G) {
        double expectedColdness = 1;

        class ModelMock {
        public:
            using VectorType = Eigen::VectorXd;
            using MatrixType = Eigen::MatrixXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 1.5);

        markovChainWithColdnessAttribute.setColdness(1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE( setColdnessToLessThan0) {
        double expectedColdness = 0;

        class ModelMock {
        public:
            using VectorType = Eigen::VectorXd;
            using MatrixType = Eigen::MatrixXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, -1.5);

        markovChainWithColdnessAttribute.setColdness(-1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        BOOST_CHECK(actualColdness == expectedColdness);
    }

    BOOST_AUTO_TEST_CASE( calculateNegativeLogLikelihood) {
        double expectedNegativeLogLikelihood = -250;

        class ModelMock {
        public:
            using VectorType = Eigen::VectorXd;
            using MatrixType = Eigen::MatrixXd;

            [[maybe_unused]] double calculateNegativeLogLikelihood(const VectorType &) {
                return -1000;
            }
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 0.25);


        hops::ColdnessAttribute<ModelMock>::VectorType mockState = Eigen::VectorXd::Zero(1);
        double actualNegativeLogLikelihood = markovChainWithColdnessAttribute.calculateNegativeLogLikelihood(mockState);
        BOOST_CHECK(actualNegativeLogLikelihood == expectedNegativeLogLikelihood);
    }

    BOOST_AUTO_TEST_SUITE_END()
