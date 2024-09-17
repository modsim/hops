#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TruncatedNormalProposalTestSuite

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <Eigen/Core>

#include "hops/FileReader/CsvReader.hpp"
#include "hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp"
#include "hops/MarkovChain/MarkovChainAdapter.hpp"
#include "hops/MarkovChain/ModelWrapper.hpp"
#include "hops/MarkovChain/Proposal/TruncatedNormalDistribution.hpp"
#include "hops/Model/Gaussian.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Statistics/EffectiveSampleSize.hpp"

BOOST_AUTO_TEST_SUITE(TruncatedNormalDistribution)

    BOOST_AUTO_TEST_CASE(TestInverseNormalizationConstant) {
        hops::TruncatedNormalDistribution<double> normal;

        // case 1: full
        double expectedInverseNormalization = 1;
        double sigma = 1;
        double lower = -std::numeric_limits<double>::infinity();
        double upper = std::numeric_limits<double>::infinity();
        double actualInverseNormalization = normal.inverseNormalization(
        {sigma, lower, upper}
        );
        BOOST_CHECK_CLOSE(actualInverseNormalization, expectedInverseNormalization, 0.01);

        // case 2: full with low sigma
        expectedInverseNormalization = 1;
        sigma = 0.0001;
        lower = -std::numeric_limits<double>::infinity();
        upper = std::numeric_limits<double>::infinity();
        actualInverseNormalization = normal.inverseNormalization(
        {sigma, lower, upper}
        );
        BOOST_CHECK_CLOSE(actualInverseNormalization, expectedInverseNormalization, 0.01);


        // case 3: half
        expectedInverseNormalization = 0.5;
        sigma = 1;
        lower = 0;
        upper = std::numeric_limits<double>::infinity();
        actualInverseNormalization = normal.inverseNormalization(
        {sigma, lower, upper}
        );
        BOOST_CHECK_CLOSE(actualInverseNormalization, expectedInverseNormalization, 0.01);

        // case 4: half
        expectedInverseNormalization = 0.5;
        sigma = 1;
        lower = -std::numeric_limits<double>::infinity();
        upper = 0;
        actualInverseNormalization = normal.inverseNormalization(
        {sigma, lower, upper}
        );
        BOOST_CHECK_CLOSE(actualInverseNormalization, expectedInverseNormalization, 0.01);

        // case 5: 1 sigma
        expectedInverseNormalization = 0.682689492137;
        sigma = 1;
        lower = -1;
        upper = 1;
        actualInverseNormalization = normal.inverseNormalization(
        {sigma, lower, upper}
        );
        BOOST_CHECK_CLOSE(actualInverseNormalization, expectedInverseNormalization, 0.01);

        // case 6: 2 sigma shifted
        expectedInverseNormalization = 0.954499736104 / 2;
        sigma = 1;
        lower = -1;
        upper = 1;
        actualInverseNormalization = normal.inverseNormalization(
        {sigma, lower-1, upper-1}
        );
        BOOST_CHECK_CLOSE(actualInverseNormalization, expectedInverseNormalization, 0.01);

        // case 7: 6 sigma shifted (4 + 6)*sigma/ 2
        expectedInverseNormalization = 0.999999998027 / 2 + 0.999936657516 / 2 ;
        sigma = 1;
        lower = -5;
        upper = 5;
        actualInverseNormalization = normal.inverseNormalization(
        {sigma, lower-1, upper-1}
        );
        BOOST_CHECK_CLOSE(actualInverseNormalization, expectedInverseNormalization, 0.01);
    }


    BOOST_AUTO_TEST_CASE(TestDraw) {
        hops::TruncatedNormalDistribution<double> normal;

        hops::RandomNumberGenerator rng(42);

        std::vector<double> result;
        double mean = 0;
        int its = 10000;
        for(int i=0; i<its; ++i) {
            double val = normal(rng, {0.01, -1, 1});
            mean += val;
            result.push_back(val);
        };
        mean /= its;
        double var =0;
        for(int i=0; i<its; ++i) {
            var += std::pow(result[i]-mean, 2);
        }
        var /= its;
        BOOST_CHECK(std::abs(std::sqrt(var)-0.01) < 0.001);
        BOOST_CHECK(std::abs(mean) < 0.001);

    }

BOOST_AUTO_TEST_SUITE_END()
