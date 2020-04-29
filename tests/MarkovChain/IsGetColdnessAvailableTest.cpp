#include <gtest/gtest.h>
#include <hops/MarkovChain/IsGetColdnessAvailable.hpp>

namespace {
    TEST(IsGetColdnessAvailable, WhenGetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsGetColdnessAvailable<Mock>::value);
    }

    TEST(IsGetColdnessAvailable, WhenGetColdnessHasWrongSignature) {
        class Mock {
        public:
            double getColdness(double);
        };

        EXPECT_FALSE(hops::IsGetColdnessAvailable<Mock>::value);
    }

    TEST(IsGetColdnessAvailable, WhenGetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double getColdness();
        };

        EXPECT_TRUE(hops::IsGetColdnessAvailable<Mock>::value);
    }
}
