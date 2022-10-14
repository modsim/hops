#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IsStoreRecordAvailableTestSuite

#include <boost/test/unit_test.hpp>

#include "hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp"

BOOST_AUTO_TEST_SUITE(IsStoreRecordAvailableTestSuite)

    BOOST_AUTO_TEST_CASE(WhenStoreRecordIsNotAvailable) {
        BOOST_CHECK(hops::IsStoreRecordAvailable<double>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenStoreRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void storeRecord(double);
        };

        BOOST_CHECK(hops::IsStoreRecordAvailable<RecorderMock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenStoreRecordIsAvailable) {
        class RecorderMock {
        public:
            void storeRecord();
        };

        BOOST_CHECK(hops::IsStoreRecordAvailable<RecorderMock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
