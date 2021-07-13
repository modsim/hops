#define BOOST_TEST_MODULE IsComputeLogLikelihoodGradientAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/Model/IsComputeLogLikelihoodGradientAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsComputeLogLikelihoodGradientAvailable)

    BOOST_AUTO_TEST_CASE(WhenComputeLogLikelihoodGradientIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsComputeLogLikelihoodGradientAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenComputeLogLikelihoodGradientHasWrongSignature) {
        class Mock {
        public:
            double computeLogLikelihoodGradient(Mock);
        };

        BOOST_CHECK(hops::IsComputeLogLikelihoodGradientAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenComputeLogLikelihoodGradientIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            using VectorType = double;

            VectorType computeLogLikelihoodGradient(const VectorType &);
        };

        BOOST_CHECK(hops::IsComputeLogLikelihoodGradientAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
