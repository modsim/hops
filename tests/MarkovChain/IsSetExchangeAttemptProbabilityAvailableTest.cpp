#include <gtest/gtest.h>
#include <hops/MarkovChain/IsSetExchangeAttemptProbabilityAvailable.hpp>

namespace {
    TEST(IsSetExchangeAttemptProbabilityAvailable, WhenSetExchangeAttemptProbabilityIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsSetExchangeAttemptProbabilityAvailable<Mock>::value);
    }

    TEST(IsSetExchangeAttemptProbabilityAvailable, WhenSetExchangeAttemptProbabilityHasWrongSignature) {
        class Mock {
        public:
            void setExchangeAttemptProbability();
        };

        EXPECT_FALSE(hops::IsSetExchangeAttemptProbabilityAvailable<Mock>::value);
    }

    TEST(IsSetExchangeAttemptProbabilityAvailable, WhenSetExchangeAttemptProbabilityIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setExchangeAttemptProbability(double);
        };

        EXPECT_TRUE(hops::IsSetExchangeAttemptProbabilityAvailable<Mock>::value);
    }
}
