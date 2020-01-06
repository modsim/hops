#include <gtest/gtest.h>
#include <nups/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>

namespace {
    TEST(IsClearRecordsAvailable, WhenRecordIsNotAvailable) {
        EXPECT_FALSE(nups::IsClearRecordsAvailable<double>::value);
    }

    TEST(IsClearRecordsAvailable, WhenRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void clearRecords(double);
        };

        EXPECT_FALSE(nups::IsClearRecordsAvailable<RecorderMock>::value);
    }

    TEST(IsClearRecordsAvailable, WhenRecordIsAvailableWithCorrectTypedef) {
        class RecorderMock {
        public:
            void clearRecords();
        };

        EXPECT_TRUE(nups::IsClearRecordsAvailable<RecorderMock>::value);
    }
}
