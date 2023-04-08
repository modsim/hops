#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ChordStepDistributionsTestSuite

#include <boost/test/unit_test.hpp>

#include "hops/MarkovChain/Proposal/ChordStepDistributions.hpp"

BOOST_AUTO_TEST_SUITE(ChordStepDistributionTestSuite)

    BOOST_AUTO_TEST_CASE(UniformStepDistributionDrawIsBounded) {
        hops::UniformStepDistribution<double> uniformStepDistribution;
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (long i = 0; i < 100'000; ++i) {
            auto lowerLimit = static_cast<double>(-i);
            auto upperLimit = static_cast<double>(i + 1);
            double draw = uniformStepDistribution.draw(randomNumberGenerator, lowerLimit, upperLimit);
            BOOST_CHECK_LT(draw, upperLimit);
            BOOST_CHECK_GT(draw, lowerLimit);
        }
    }

    BOOST_AUTO_TEST_CASE(UniformStepDistributionInverseNormalizationIsCorrect) {
        hops::UniformStepDistribution<double> uniformStepDistribution;
        double actualInverseNormalization = uniformStepDistribution.computeInverseNormalizationConstant(1,
                                                                                                        5,
                                                                                                        9);
        BOOST_CHECK_EQUAL(actualInverseNormalization, 1);
    }

    BOOST_AUTO_TEST_CASE(UniformStepDistributionThrowsWhenUpperLimitIsInfinity) {
        hops::UniformStepDistribution<double> uniformStepDistribution;
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

        BOOST_CHECK_THROW(
                uniformStepDistribution.draw(randomNumberGenerator, 0, std::numeric_limits<double>::infinity()),
                std::invalid_argument
        );
    }

    BOOST_AUTO_TEST_CASE(UniformStepDistributionThrowsWhenLowerLimitIsMinusInfinity) {
        hops::UniformStepDistribution<double> uniformStepDistribution;
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

        BOOST_CHECK_THROW(
                uniformStepDistribution.draw(randomNumberGenerator, -std::numeric_limits<double>::infinity(), 10),
                std::invalid_argument
        );
    }

    BOOST_AUTO_TEST_CASE(UniformStepDistributionThrowsWhenLowerLimitIsLargerThanUpperLimit) {
        hops::UniformStepDistribution<double> uniformStepDistribution;
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

        BOOST_CHECK_THROW(
                uniformStepDistribution.draw(randomNumberGenerator, 1, -1),
                std::invalid_argument
        );
    }

    BOOST_AUTO_TEST_CASE(GaussianStepDistributionDrawIsBounded) {
        hops::GaussianStepDistribution<double> gaussianStepDistribution;

        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (long i = 0; i < 100000; ++i) {
            auto lowerLimit = static_cast<double>(-i);
            auto upperLimit = static_cast<double>(i + 1);
            double draw = gaussianStepDistribution.draw(randomNumberGenerator, lowerLimit, upperLimit);
            BOOST_CHECK_LT(draw, upperLimit);
            BOOST_CHECK_GT(draw, lowerLimit);
        }
    }

    BOOST_AUTO_TEST_CASE(GaussianStepDistributionDrawIsBoundedWhenSigmaIsLarge) {
        hops::GaussianStepDistribution<double> gaussianStepDistribution;
        gaussianStepDistribution.setStepSize(10000);

        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (long i = 0; i < 100000; ++i) {
            auto lowerLimit = static_cast<double>(-1);
            auto upperLimit = static_cast<double>(1);
            double draw = gaussianStepDistribution.draw(randomNumberGenerator, lowerLimit, upperLimit);
            BOOST_CHECK_LT(draw, upperLimit);
            BOOST_CHECK_GT(draw, lowerLimit);
        }
    }

    BOOST_AUTO_TEST_CASE(GaussianStepDistributionInverseNormilizationIsCorrect) {
        hops::GaussianStepDistribution<double> gaussianStepDistribution;
        double actualInverseNormalization = gaussianStepDistribution.computeInverseNormalizationConstant(.5,
                                                                                                         -.5,
                                                                                                         .9);
        BOOST_CHECK_SMALL(actualInverseNormalization - 0.805414, 1e-3);
    }


    BOOST_AUTO_TEST_CASE(GaussianStepDistributionThrowsWhenLowerLimitIsLargerThanUpperLimit) {
        hops::GaussianStepDistribution<double> gaussianStepDistribution;
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

        BOOST_CHECK_THROW(
                gaussianStepDistribution.draw(randomNumberGenerator, 1, -1),
                std::invalid_argument
        );


        BOOST_CHECK_THROW(
                gaussianStepDistribution.draw(randomNumberGenerator, 0, 1, -1),
                std::invalid_argument
        );

        BOOST_CHECK_THROW(
                gaussianStepDistribution.computeInverseNormalizationConstant(0, 1, -1),
                std::invalid_argument
        );

        BOOST_CHECK_THROW(
                gaussianStepDistribution.computeProbabilityDensity(0, 0, 1, -1),
                std::invalid_argument
        );
    }
}

