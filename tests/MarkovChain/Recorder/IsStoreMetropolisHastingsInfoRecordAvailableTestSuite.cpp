#define BOOST_TEST_MODULE IsStoreMetropolisHastingsInfoRecordAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/Recorder/IsStoreMetropolisHastingsInfoRecordAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsStoreMetropolisHastingsInfoRecordAvailableTestSuite)

    BOOST_AUTO_TEST_CASE(WhenSetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(!hops::IsStoreMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenSetColdnessHasWrongSignature) {
        class Mock {
        public:
            void storeMetropolisHastingsInfoRecord();
        };

        BOOST_CHECK(!hops::IsStoreMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenSetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void storeMetropolisHastingsInfoRecord(const std::string &);
        };

        BOOST_CHECK(hops::IsStoreMetropolisHastingsInfoRecordAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
