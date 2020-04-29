#include <gtest/gtest.h>
#include <hops/MarkovChain/IsGetStepSizeAvailable.hpp>

namespace {
    TEST(IsGetStepSizeAvailable, WhenGetStepSizeIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsGetStepSizeAvailable<Mock>::value);
    }

    TEST(IsGetStepSizeAvailable, WhenGetStepSizeHasWrongSignature) {
        class Mock {
        public:
            double getStepSize(double);
        };

        EXPECT_FALSE(hops::IsGetStepSizeAvailable<Mock>::value);
    }

    TEST(IsGetStepSizeAvailable, WhenGetStepSizeIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double getStepSize();
        };

        EXPECT_TRUE(hops::IsGetStepSizeAvailable<Mock>::value);
    }
}
