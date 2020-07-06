#include <gtest/gtest.h>
#include <hops/MarkovChain/IsSetColdnessAvailable.hpp>

namespace {
    TEST(IsSetColdnessAvailable, WhenSetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsSetColdnessAvailable<Mock>::value);
    }

    TEST(IsSetColdnessAvailable, WhenSetColdnessHasWrongSignature) {
        class Mock {
        public:
            void setColdness();
        };

        EXPECT_FALSE(hops::IsSetColdnessAvailable<Mock>::value);
    }

    TEST(IsSetColdnessAvailable, WhenSetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setColdness(double);
        };

        EXPECT_TRUE(hops::IsSetColdnessAvailable<Mock>::value);
    }
}
