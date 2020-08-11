#include <gtest/gtest.h>
#include <hops/MarkovChain/Recorder/IsStoreMetropolisHastingsInfoRecordAvailable.hpp>

namespace {
    TEST(IsStoreMetropolisHastingsInfoRecordAvailable, WhenSetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsStoreMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    TEST(IsStoreMetropolisHastingsInfoRecordAvailable, WhenSetColdnessHasWrongSignature) {
        class Mock {
        public:
            void storeMetropolisHastingsInfoRecord();
        };

        EXPECT_FALSE(hops::IsStoreMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    TEST(IsStoreMetropolisHastingsInfoRecordAvailable, WhenSetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void storeMetropolisHastingsInfoRecord(const std::string &);
        };

        EXPECT_TRUE(hops::IsStoreMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }
}
