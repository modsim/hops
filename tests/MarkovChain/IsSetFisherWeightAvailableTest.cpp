#include <gtest/gtest.h>
#include <hops/MarkovChain/IsSetFisherWeightAvailable.hpp>

namespace {
    TEST(IsSetFisherWeightAvailable, WhenSetFisherWeightIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsSetFisherWeightAvailable<Mock>::value);
    }

    TEST(IsSetFisherWeightAvailable, WhenSetFisherWeightHasWrongSignature) {
        class Mock {
        public:
            void setFisherWeight();
        };

        EXPECT_FALSE(hops::IsSetFisherWeightAvailable<Mock>::value);
    }

    TEST(IsSetFisherWeightAvailable, WhenSetFisherWeightIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setFisherWeight(double);
        };

        EXPECT_TRUE(hops::IsSetFisherWeightAvailable<Mock>::value);
    }
}
