#define BOOST_TEST_MODULE IsCalculateLogLikelihoodGradientAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/Model/IsCalculateLogLikelihoodGradientAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsCalculateLogLikelihoodGradientAvailable)

    BOOST_AUTO_TEST_CASE(WhenCalculateLogLikelihoodGradientIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsCalculateLogLikelihoodGradientAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenCalculateLogLikelihoodGradientHasWrongSignature) {
        class Mock {
        public:
            double calculateLogLikelihoodGradient(Mock);
        };

        BOOST_CHECK(hops::IsCalculateLogLikelihoodGradientAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenCalculateLogLikelihoodGradientIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            using VectorType = double;

            VectorType calculateLogLikelihoodGradient(const VectorType &);
        };

        BOOST_CHECK(hops::IsCalculateLogLikelihoodGradientAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
