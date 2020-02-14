#include <gtest/gtest.h>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>

namespace {
    TEST(IsStoreRecordAvailable, WhenStoreRecordIsNotAvailable) {
        EXPECT_FALSE(hops::IsWriteRecordsToFileAvailable<double>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void writeRecordsToFile();
        };

        EXPECT_FALSE(hops::IsWriteRecordsToFileAvailable<RecorderMock>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordIsAvailable) {
        class RecorderMock {
        public:
            void writeRecordsToFile(const hops::FileWriter *);
        };

        EXPECT_TRUE(hops::IsWriteRecordsToFileAvailable<RecorderMock>::value);
    }
}
