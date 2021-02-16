#define BOOST_TEST_MODULE IsGetExchangeAttemptProbabilityAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/IsGetExchangeAttemptProbabilityAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsGetExchangeAttemptProbabilityAvailable)

    BOOST_AUTO_TEST_CASE( WhenGetExchangeAttemptProbabilityIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsGetExchangeAttemptProbabilityAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenGetExchangeAttemptProbabilityHasWrongSignature) {
        class Mock {
        public:
            double getExchangeAttemptProbability(double);
        };

        BOOST_CHECK(hops::IsGetExchangeAttemptProbabilityAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenGetExchangeAttemptProbabilityIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            double getExchangeAttemptProbability();
        };

        BOOST_CHECK(hops::IsGetExchangeAttemptProbabilityAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
