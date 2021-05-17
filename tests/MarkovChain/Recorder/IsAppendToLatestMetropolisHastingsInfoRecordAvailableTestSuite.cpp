#define BOOST_TEST_MODULE IsAppendToLatestMetropolisHastingsInfoRecordAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/Recorder/IsAppendToLatestMetropolisHastingsInfoRecordAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsAppendToLatestMetropolisHastingsInfoRecordAvailableTestSuite)
    BOOST_AUTO_TEST_CASE(WhenSetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(!hops::IsAppendToLatestMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenSetColdnessHasWrongSignature) {
        class Mock {
        public:
            void appendToLatestMetropolisHastingsInfoRecord();
        };

        BOOST_CHECK(!hops::IsAppendToLatestMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenSetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void appendToLatestMetropolisHastingsInfoRecord(const std::string&);
        };

        BOOST_CHECK(hops::IsAppendToLatestMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }
BOOST_AUTO_TEST_SUITE_END()
