#include <gtest/gtest.h>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>

namespace {
    TEST(IsClearRecordsAvailable, WhenRecordIsNotAvailable) {
        EXPECT_FALSE(hops::IsClearRecordsAvailable<double>::value);
    }

    TEST(IsClearRecordsAvailable, WhenRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void clearRecords(double);
        };

        EXPECT_FALSE(hops::IsClearRecordsAvailable<RecorderMock>::value);
    }

    TEST(IsClearRecordsAvailable, WhenRecordIsAvailableWithCorrectTypedef) {
        class RecorderMock {
        public:
            void clearRecords();
        };

        EXPECT_TRUE(hops::IsClearRecordsAvailable<RecorderMock>::value);
    }
}
