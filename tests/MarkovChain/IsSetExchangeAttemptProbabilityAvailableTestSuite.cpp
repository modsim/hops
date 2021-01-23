#define BOOST_TEST_MODULE IsSetExchangeAttemptProbabilityAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/IsSetExchangeAttemptProbabilityAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsSetExchangeAttemptProbabilityAvailable)

    BOOST_AUTO_TEST_CASE( WhenSetExchangeAttemptProbabilityIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsSetExchangeAttemptProbabilityAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenSetExchangeAttemptProbabilityHasWrongSignature) {
        class Mock {
        public:
            void setExchangeAttemptProbability();
        };

        BOOST_CHECK(hops::IsSetExchangeAttemptProbabilityAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenSetExchangeAttemptProbabilityIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setExchangeAttemptProbability(double);
        };

        BOOST_CHECK(hops::IsSetExchangeAttemptProbabilityAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_SUITE_END()
