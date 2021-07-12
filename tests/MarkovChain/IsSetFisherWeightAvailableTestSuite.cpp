#define BOOST_TEST_MODULE IsSetFisherWeightAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/IsSetFisherWeightAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsSetFisherWeightAvailableTestSuite)

    BOOST_AUTO_TEST_CASE(WhenSetFisherWeightIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(!hops::IsSetFisherWeightAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenSetFisherWeightHasWrongSignature) {
        class Mock {
        public:
            void setFisherWeight();
        };

        BOOST_CHECK(!hops::IsSetFisherWeightAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenSetFisherWeightIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setFisherWeight(double);
        };

        BOOST_CHECK(hops::IsSetFisherWeightAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
