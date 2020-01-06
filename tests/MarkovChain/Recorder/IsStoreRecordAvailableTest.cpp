#include <gtest/gtest.h>
#include <nups/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>

namespace {
    TEST(IsStoreRecordAvailable, WhenStoreRecordIsNotAvailable) {
        EXPECT_FALSE(nups::IsStoreRecordAvailable<double>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void storeRecord(double);
        };

        EXPECT_FALSE(nups::IsStoreRecordAvailable<RecorderMock>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordIsAvailable) {
        class RecorderMock {
        public:
            void storeRecord();
        };

        EXPECT_TRUE(nups::IsStoreRecordAvailable<RecorderMock>::value);
    }
}
