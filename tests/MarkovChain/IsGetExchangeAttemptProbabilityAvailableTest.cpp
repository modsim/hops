#include <gtest/gtest.h>
#include <hops/MarkovChain/IsGetExchangeAttemptProbabilityAvailable.hpp>

namespace {
    TEST(IsGetExchangeAttemptProbabilityAvailable, WhenGetExchangeAttemptProbabilityIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsGetExchangeAttemptProbabilityAvailable<Mock>::value);
    }

    TEST(IsGetExchangeAttemptProbabilityAvailable, WhenGetExchangeAttemptProbabilityHasWrongSignature) {
        class Mock {
        public:
            double getExchangeAttemptProbability(double);
        };

        EXPECT_FALSE(hops::IsGetExchangeAttemptProbabilityAvailable<Mock>::value);
    }

    TEST(IsGetExchangeAttemptProbabilityAvailable, WhenGetExchangeAttemptProbabilityIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double getExchangeAttemptProbability();
        };

        EXPECT_TRUE(hops::IsGetExchangeAttemptProbabilityAvailable<Mock>::value);
    }
}
