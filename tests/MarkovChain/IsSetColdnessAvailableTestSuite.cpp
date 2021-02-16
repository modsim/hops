#define BOOST_TEST_MODULE IsSetColdnessAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/IsSetColdnessAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsSetColdnessAvailable)

    BOOST_AUTO_TEST_CASE( WhenSetColdnessIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsSetColdnessAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenSetColdnessHasWrongSignature) {
        class Mock {
        public:
            void setColdness();
        };

        BOOST_CHECK(hops::IsSetColdnessAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE( WhenSetColdnessIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setColdness(double);
        };

        BOOST_CHECK(hops::IsSetColdnessAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
