#include <gtest/gtest.h>
#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>

namespace {
    TEST(IsAddMessageAvailable, WhenAddMessageIsNotAvailable) {
        class Mock {
        public:
        };
        EXPECT_FALSE(hops::IsAddMessageAvailable<Mock>::value);
    }

    TEST(IsAddMessageAvailable, WhenAddMessageHasWrongSignature) {
        class Mock {
        public:
            void addMessage();
        };

        EXPECT_FALSE(hops::IsAddMessageAvailable<Mock>::value);
    }

    TEST(IsAddMessageAvailable, WhenAddMessageIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void addMessage(const std::string &);
        };

        EXPECT_TRUE(hops::IsAddMessageAvailable<Mock>::value);
    }
}
