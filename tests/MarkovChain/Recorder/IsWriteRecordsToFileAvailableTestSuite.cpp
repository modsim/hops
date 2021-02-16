#define BOOST_TEST_MODULE IsWriteRecordsToFileAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsWriteRecordsToFileAvailable)

    BOOST_AUTO_TEST_CASE(WhenStoreRecordIsNotAvailable) {
        BOOST_CHECK(hops::IsWriteRecordsToFileAvailable<double>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenStoreRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void writeRecordsToFile();
        };

        BOOST_CHECK(hops::IsWriteRecordsToFileAvailable<RecorderMock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenStoreRecordIsAvailable) {
        class RecorderMock {
        public:
            void writeRecordsToFile(const hops::FileWriter *);
        };

        BOOST_CHECK(hops::IsWriteRecordsToFileAvailable<RecorderMock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
