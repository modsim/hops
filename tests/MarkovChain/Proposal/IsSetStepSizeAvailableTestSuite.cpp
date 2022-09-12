#define BOOST_TEST_MODULE IsSetStepSizeAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/Proposal/IsSetStepSizeAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsSetStepSizeAvailable)

    BOOST_AUTO_TEST_CASE(WhenSetStepSizeIsNotAvailable) {
        class Mock {
        public:
        };
        bool isStepSizeAvailable = hops::IsSetStepSizeAvailable<Mock>::value;
        BOOST_CHECK(isStepSizeAvailable == false);
    }

    BOOST_AUTO_TEST_CASE(WhenSetStepSizeHasWrongSignature) {
        class Mock {
        public:
            void setStepSize();
        };

        bool isStepSizeAvailable = hops::IsSetStepSizeAvailable<Mock>::value;
        BOOST_CHECK(isStepSizeAvailable == false);
    }

    BOOST_AUTO_TEST_CASE(WhenSetStepSizeIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void setStepSize(float);
        };

        bool isStepSizeAvailable = hops::IsSetStepSizeAvailable<Mock>::value;
        BOOST_CHECK(isStepSizeAvailable);
    }

BOOST_AUTO_TEST_SUITE_END()
