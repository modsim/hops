#include <gtest/gtest.h>
#include <hops/MarkovChain/ParallelTempering/ColdnessAttribute.hpp>

namespace {
    TEST(ColdnessAttribute, getColdness) {
        double expectedColdness = 0.5;
        
        class MarkovChainMock {
        public:
            using StateType=double;
        };

        hops::ColdnessAttribute<MarkovChainMock> markovChainWithColdnessAttribute(markovChainWithColdnessAttribute,
                                                                                  expectedColdness);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, setColdness) {
        double expectedColdness = 0.5;

        class MarkovChainMock {
        public:
            using StateType=double;
        };

        hops::ColdnessAttribute<MarkovChainMock> markovChainWithColdnessAttribute(markovChainWithColdnessAttribute,
                                                                                  1.);

        markovChainWithColdnessAttribute.setColdness(expectedColdness);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, constructorWithColdnessLargerThan1) {
        double expectedColdness = 1;

        class MarkovChainMock {
        public:
            using StateType=double;
        };

        hops::ColdnessAttribute<MarkovChainMock> markovChainWithColdnessAttribute(markovChainWithColdnessAttribute,
                                                                                  1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, constructorWithColdnessLessThan0) {
        double expectedColdness = 0;

        class MarkovChainMock {
        public:
            using StateType=double;
        };

        hops::ColdnessAttribute<MarkovChainMock> markovChainWithColdnessAttribute(markovChainWithColdnessAttribute,
                                                                                  -1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, setColdnessToMoreThan1G) {
        double expectedColdness = 1;

        class MarkovChainMock {
        public:
            using StateType=double;
        };

        hops::ColdnessAttribute<MarkovChainMock> markovChainWithColdnessAttribute(markovChainWithColdnessAttribute,
                                                                                  1.5);

        markovChainWithColdnessAttribute.setColdness(1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, setColdnessToLessThan0) {
        double expectedColdness = 0;

        class MarkovChainMock {
        public:
            using StateType=double;
        };

        hops::ColdnessAttribute<MarkovChainMock> markovChainWithColdnessAttribute(markovChainWithColdnessAttribute,
                                                                                  -1.5);

        markovChainWithColdnessAttribute.setColdness(-1.5);

        double actualColdness = markovChainWithColdnessAttribute.getColdness();
        EXPECT_EQ(actualColdness, expectedColdness);
    }

    TEST(ColdnessAttribute, calculateNegativeLogLikelihood) {
        double expectedNegativeLogLikelihood = -250;

        class MarkovChainMock {
        public:
            using StateType=double;
            double calculateNegativeLogLikelihood(const StateType&) {
                return -1000;
            }
        };

        hops::ColdnessAttribute<MarkovChainMock> markovChainWithColdnessAttribute(markovChainWithColdnessAttribute,
                                                                                  0.25);


        hops::ColdnessAttribute<MarkovChainMock>::StateType mockState = 0;
        double actualNegativeLogLikelihood = markovChainWithColdnessAttribute.calculateNegativeLogLikelihood(mockState);
        EXPECT_EQ(actualNegativeLogLikelihood, expectedNegativeLogLikelihood);
    }
}
