#include <gtest/gtest.h>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>

namespace {
    namespace UniformStepDistribution {
        TEST(UniformStepDistribution, DrawIsBounded) {
            hops::UniformStepDistribution<double> uniformStepDistribution;
            hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
            for (long i = 0; i < 100000; ++i) {
                auto lowerLimit = static_cast<double>(-i);
                auto upperLimit = static_cast<double>(i + 1);
                double draw = uniformStepDistribution.draw(randomNumberGenerator, lowerLimit, upperLimit);
                EXPECT_LE(draw, upperLimit);
                EXPECT_GE(draw, lowerLimit);
            }
        }

        TEST(UniformStepDistribution, InverseNormilizationIsCorrect) {
            hops::UniformStepDistribution<double> uniformStepDistribution;
            double actualInverseNormalization = uniformStepDistribution.calculateInverseNormalizationConstant(1,
                                                                                                              5,
                                                                                                              9);
            EXPECT_DOUBLE_EQ(actualInverseNormalization, 1);
        }
    }

    namespace GaussianStepDistribution {
        TEST(UniformStepDistribution, DrawIsBounded) {
            hops::GaussianStepDistribution<double> gaussianStepDistribution;

            hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
            for (long i = 0; i < 100000; ++i) {
                auto lowerLimit = static_cast<double>(-i);
                auto upperLimit = static_cast<double>(i + 1);
                double draw = gaussianStepDistribution.draw(randomNumberGenerator, lowerLimit, upperLimit);
                EXPECT_LE(draw, upperLimit);
                EXPECT_GE(draw, lowerLimit);
            }
        }

        TEST(UniformStepDistribution, DrawIsBoundedWhenSigmaIsLarge) {
            hops::GaussianStepDistribution<double> gaussianStepDistribution;
            gaussianStepDistribution.setStepSize(10000);

            hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
            for (long i = 0; i < 100000; ++i) {
                auto lowerLimit = static_cast<double>(-1);
                auto upperLimit = static_cast<double>(1);
                double draw = gaussianStepDistribution.draw(randomNumberGenerator, lowerLimit, upperLimit);
                EXPECT_LE(draw, upperLimit);
                EXPECT_GE(draw, lowerLimit);
            }
        }

        TEST(UniformStepDistribution, InverseNormilizationIsCorrect) {
            hops::GaussianStepDistribution<double> gaussianStepDistribution;
            double actualInverseNormalization = gaussianStepDistribution.calculateInverseNormalizationConstant(.5,
                                                                                                               -.5,
                                                                                                               .9);
            EXPECT_NEAR(actualInverseNormalization, 0.805414, 1e-3);
        }
    }
}

