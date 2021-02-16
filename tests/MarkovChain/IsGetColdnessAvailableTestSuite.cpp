#define BOOST_TEST_MODULE IsGetColdnessAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/IsGetColdnessAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsGetColdnessAvailable)
    BOOST_AUTO_TEST_CASE( WhenGetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsGetColdnessAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenGetColdnessHasWrongSignature) {
        class Mock {
        public:
            double getColdness(double);
        };

        BOOST_CHECK(hops::IsGetColdnessAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenGetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double getColdness();
        };

        BOOST_CHECK(hops::IsGetColdnessAvailable<Mock>::value);
    }
BOOST_AUTO_TEST_SUITE_END()
