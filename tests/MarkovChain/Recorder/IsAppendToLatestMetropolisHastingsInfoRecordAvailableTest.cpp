#include <gtest/gtest.h>
#include <hops/MarkovChain/Recorder/IsAppendToLatestMetropolisHastingsInfoRecordAvailable.hpp>

namespace {
    TEST(IsAppendToLatestMetropolisHastingsInfoRecordAvailable, WhenSetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsAppendToLatestMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    TEST(IsAppendToLatestMetropolisHastingsInfoRecordAvailable, WhenSetColdnessHasWrongSignature) {
        class Mock {
        public:
            void appendToLatestMetropolisHastingsInfoRecord();
        };

        EXPECT_FALSE(hops::IsAppendToLatestMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    TEST(IsAppendToLatestMetropolisHastingsInfoRecordAvailable, WhenSetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void appendToLatestMetropolisHastingsInfoRecord(const std::string&);
        };

        EXPECT_TRUE(hops::IsAppendToLatestMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }
}
