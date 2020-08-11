#include <gtest/gtest.h>
#include <hops/MarkovChain/IsResetAcceptanceRateAvailable.hpp>

namespace {
    TEST(IsResetAcceptanceRateAvailable, WhenResetAcceptanceRateIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsResetAcceptanceRateAvailable<Mock>::value);
    }

    TEST(IsResetAcceptanceRateAvailable, WhenResetAcceptanceRateHasWrongSignature) {
        class Mock {
        public:
            double resetAcceptanceRate(double);
        };

        EXPECT_FALSE(hops::IsResetAcceptanceRateAvailable<Mock>::value);
    }

    TEST(IsResetAcceptanceRateAvailable, WhenResetAcceptanceRateIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double resetAcceptanceRate();
        };

        EXPECT_TRUE(hops::IsResetAcceptanceRateAvailable<Mock>::value);
    }
}
