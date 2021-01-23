#define BOOST_TEST_MODULE IsGetStepSizeAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/IsGetStepSizeAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsGetStepSizeAvailable)

    BOOST_AUTO_TEST_CASE( WhenGetStepSizeIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsGetStepSizeAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenGetStepSizeHasWrongSignature) {
        class Mock {
        public:
            double getStepSize(double);
        };

        BOOST_CHECK(hops::IsGetStepSizeAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenGetStepSizeIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double getStepSize();
        };

        BOOST_CHECK(hops::IsGetStepSizeAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
