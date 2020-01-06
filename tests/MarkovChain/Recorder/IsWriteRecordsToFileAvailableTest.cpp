#include <gtest/gtest.h>
#include <nups/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>

namespace {
    TEST(IsStoreRecordAvailable, WhenStoreRecordIsNotAvailable) {
        EXPECT_FALSE(nups::IsWriteRecordsToFileAvailable<double>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void writeRecordsToFile();
        };

        EXPECT_FALSE(nups::IsWriteRecordsToFileAvailable<RecorderMock>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordIsAvailable) {
        class RecorderMock {
        public:
            void writeRecordsToFile(const nups::FileWriter *);
        };

        EXPECT_TRUE(nups::IsWriteRecordsToFileAvailable<RecorderMock>::value);
    }
}
