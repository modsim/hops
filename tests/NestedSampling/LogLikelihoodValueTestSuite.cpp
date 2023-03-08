#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LogLikelihoodValueTestSuite

#include <boost/test/unit_test.hpp>
#include "hops/NestedSampling/LogLikelihoodValue.hpp"

BOOST_AUTO_TEST_SUITE(LogLikelihoodValueTestSuite)

BOOST_AUTO_TEST_CASE(TestComparisons) {

    hops::LogLikelihoodValue smallValue(5, 0.1);
    hops::LogLikelihoodValue largeValue(50, 0);
    hops::LogLikelihoodValue equalButLargerValue(5, 0.2);

    BOOST_CHECK(smallValue < largeValue);
    BOOST_CHECK(!(largeValue < smallValue));
    BOOST_CHECK(smallValue < equalButLargerValue);
    BOOST_CHECK(!(equalButLargerValue < smallValue));
}

BOOST_AUTO_TEST_SUITE_END()

