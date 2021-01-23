#define BOOST_TEST_MODULE IsClearRecordsAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsClearRecordsAvailable)

    BOOST_AUTO_TEST_CASE( WhenRecordIsNotAvailable) {
        BOOST_CHECK(hops::IsClearRecordsAvailable<double>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenRecordHasWrongSignature) {
        class RecorderMock {
        public:
            void clearRecords(double);
        };

        BOOST_CHECK(hops::IsClearRecordsAvailable<RecorderMock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenRecordIsAvailableWithCorrectTypedef) {
        class RecorderMock {
        public:
            void clearRecords();
        };

        BOOST_CHECK(hops::IsClearRecordsAvailable<RecorderMock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
