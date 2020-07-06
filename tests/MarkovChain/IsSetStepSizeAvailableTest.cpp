#include <gtest/gtest.h>
#include <hops/MarkovChain/IsSetStepSizeAvailable.hpp>

namespace {
    TEST(IsSetStepSizeAvailable, WhenSetStepSizeIsNotAvailable) {
        class Mock {
        public:
        };
        bool isStepSizeAvailable = hops::IsSetStepSizeAvailable<Mock>::value;
        EXPECT_FALSE(isStepSizeAvailable);
    }

    TEST(IsSetStepSizeAvailable, WhenSetStepSizeHasWrongSignature) {
        class Mock {
        public:
            void setStepSize();
        };

        bool isStepSizeAvailable = hops::IsSetStepSizeAvailable<Mock>::value;
        EXPECT_FALSE(isStepSizeAvailable);
    }

    TEST(IsSetStepSizeAvailable, WhenSetStepSizeIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setStepSize(float);
        };

        bool isStepSizeAvailable = hops::IsSetStepSizeAvailable<Mock>::value;
        EXPECT_TRUE(isStepSizeAvailable);
    }
}
