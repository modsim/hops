#define BOOST_TEST_MODULE IsResetAcceptanceRateAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/IsResetAcceptanceRateAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsResetAcceptanceRateAvailableTestSuite)

    BOOST_AUTO_TEST_CASE(WhenResetAcceptanceRateIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(!hops::IsResetAcceptanceRateAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenResetAcceptanceRateHasWrongSignature) {
        class Mock {
        public:
            double resetAcceptanceRate(double);
        };

        BOOST_CHECK(!hops::IsResetAcceptanceRateAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenResetAcceptanceRateIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double resetAcceptanceRate();
        };

        BOOST_CHECK(hops::IsResetAcceptanceRateAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
