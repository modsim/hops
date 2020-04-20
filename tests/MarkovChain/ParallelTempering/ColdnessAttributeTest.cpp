#include <gtest/gtest.h>
#include <hops/MarkovChain/ParallelTempering/ColdnessAttribute.hpp>
#include <Eigen/Core>

namespace {
    TEST(ColdnessAttribute, getColdness) {
        double expectedColdness = 0.5;

        class ModelMock {
        public:
            using MatrixType = Eigen::MatrixXd;
            using VectorType = Eigen::VectorXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, expectedColdness);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, setColdness) {
        double expectedColdness = 0.5;

        class ModelMock {
        public:
            using MatrixType = Eigen::MatrixXd;
            using VectorType = Eigen::VectorXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 1.);

        markovChainWithColdnessAttribute.setColdness(expectedColdness);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, constructorWithColdnessLargerThan1) {
        double expectedColdness = 1;

        class ModelMock {
        public:
            using MatrixType = Eigen::MatrixXd;
            using VectorType = Eigen::VectorXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, constructorWithColdnessLessThan0) {
        double expectedColdness = 0;

        class ModelMock {
        public:
            using MatrixType = Eigen::MatrixXd;
            using VectorType = Eigen::VectorXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, -1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, setColdnessToMoreThan1G) {
        double expectedColdness = 1;

        class ModelMock {
        public:
            using MatrixType = Eigen::MatrixXd;
            using VectorType = Eigen::VectorXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 1.5);

        markovChainWithColdnessAttribute.setColdness(1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, setColdnessToLessThan0) {
        double expectedColdness = 0;

        class ModelMock {
        public:
            using MatrixType = Eigen::MatrixXd;
            using VectorType = Eigen::VectorXd;
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, -1.5);

        markovChainWithColdnessAttribute.setColdness(-1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, calculateNegativeLogLikelihood) {
        double expectedNegativeLogLikelihood = -250;

        class ModelMock {
        public:
            using MatrixType = Eigen::MatrixXd;
            using VectorType = Eigen::VectorXd;

            double calculateNegativeLogLikelihood(const VectorType &) {
                return -1000;
            }
        } modelMock;

        hops::ColdnessAttribute markovChainWithColdnessAttribute(modelMock, 0.25);


        hops::ColdnessAttribute<ModelMock>::VectorType mockState = Eigen::VectorXd::Zero(1);
        double actualNegativeLogLikelihood = markovChainWithColdnessAttribute.calculateNegativeLogLikelihood(mockState);
        EXPECT_EQ(actualNegativeLogLikelihood, expectedNegativeLogLikelihood);
    }
}
