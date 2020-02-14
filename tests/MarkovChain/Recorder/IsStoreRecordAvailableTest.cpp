#include <gtest/gtest.h>
#include <hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>

namespace {
    TEST(IsStoreRecordAvailable, WhenStoreRecordIsNotAvailable) {
        EXPECT_FALSE(hops::IsStoreRecordAvailable<double>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void storeRecord(double);
        };

        EXPECT_FALSE(hops::IsStoreRecordAvailable<RecorderMock>::value);
    }

    TEST(IsStoreRecordAvailable, WhenStoreRecordIsAvailable) {
        class RecorderMock {
        public:
            void storeRecord();
        };

        EXPECT_TRUE(hops::IsStoreRecordAvailable<RecorderMock>::value);
    }
}
