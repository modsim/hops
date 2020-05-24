#include <gtest/gtest.h>
#include <Eigen/Core>
#include <hops/MarkovChain/MarkovChainFactory.hpp>

namespace {
    class ModelMock {
    public:
        using VectorType = Eigen::VectorXd;

        [[maybe_unused]] double calculateNegativeLogLikelihood(const VectorType &) {
            return 0.;
        }

        [[maybe_unused]] Eigen::VectorXd calculateLogLikelihoodGradient(const VectorType &) {
            return Eigen::VectorXd::Ones(2);
        }

        [[maybe_unused]] Eigen::MatrixXd calculateExpectedFisherInformation(const VectorType &) {
            return Eigen::MatrixXd::Identity(2, 2);
        }
    };

    /**
     * @brief Fixture for tests.
     */
    class MarkovChainFactory : public ::testing::Test {
    public:
        void SetUp() override {
            A = Eigen::MatrixXd(4, 2);
            A << Eigen::MatrixXd::Identity(2, 2), -Eigen::MatrixXd::Identity(2, 2);
            b = Eigen::VectorXd::Ones(4);
            startingPoint = 0.5*Eigen::VectorXd::Ones(2);
            N = Eigen::MatrixXd::Identity(2, 2);
            shift = Eigen::VectorXd::Ones(2);
        }

        Eigen::MatrixXd A, N;
        Eigen::VectorXd b, startingPoint, shift;
        ModelMock model;
    };

    TEST_F(MarkovChainFactory, createUniformCoordinateHitAndRun) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        A,
                        b,
                        startingPoint)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createUniformCSmMALA) {
        EXPECT_THROW(
                hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CSmMALA, A, b, startingPoint),
                std::runtime_error
        );
    }

    TEST_F(MarkovChainFactory, createUniformDikinWalk) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        A,
                        b,
                        startingPoint)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createUniformHitAndRun) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        A,
                        b,
                        startingPoint)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createUniformCoordinateHitAndRunRounded) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createUniformCSmMALARounded) {
        EXPECT_THROW(
                hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CSmMALA, A, b, startingPoint, N, shift),
                std::runtime_error
        );
    }

    TEST_F(MarkovChainFactory, createUniformDikinWalkRounded) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createUniformHitAndRunRounded) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformCoordinateHitAndRun) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        A,
                        b,
                        startingPoint,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformCSmMALA) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CSmMALA, A, b, startingPoint, model, useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformDikinWalk) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        A,
                        b,
                        startingPoint,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformHitAndRun) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        A,
                        b,
                        startingPoint,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformCoordinateHitAndRunRounded) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformCSmMALARounded) {
        bool useParallelTempering = false;

        EXPECT_THROW(
                hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CSmMALA, A, b, startingPoint, N, shift, model, useParallelTempering),
                std::runtime_error
        );
    }

    TEST_F(MarkovChainFactory, createNonUniformDikinWalkRounded) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformHitAndRunRounded) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

#ifdef HOPS_MPI_SUPPORTED

    TEST_F(MarkovChainFactory, createNonUniformCoordinateHitAndRunWithParallelTempering) {
        bool useParallelTempering = true;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        A,
                        b,
                        startingPoint,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformCSmMALAWithParallelTempering) {
        bool useParallelTempering = true;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CSmMALA, A, b, startingPoint, model, useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformDikinWalkWithParallelTempering) {
        bool useParallelTempering = true;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        A,
                        b,
                        startingPoint,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformHitAndRunWithParallelTempering) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        A,
                        b,
                        startingPoint,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }
    TEST_F(MarkovChainFactory, createNonUniformCoordinateHitAndRunRoundedWithParallelTempering) {
        bool useParallelTempering = true;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformCSmMALARoundedWithParallelTempering) {
        ModelMock model;
        bool useParallelTempering = true;

        EXPECT_THROW(
                hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CSmMALA, A, b, startingPoint, N, shift, model, useParallelTempering),
                std::runtime_error
        );
    }

    TEST_F(MarkovChainFactory, createNonUniformDikinWalkRoundedWithParallelTempering) {
        bool useParallelTempering = true;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }

    TEST_F(MarkovChainFactory, createNonUniformHitAndRunRoundedWithParallelTempering) {
        bool useParallelTempering = false;

        std::unique_ptr<hops::MarkovChain> markovChain;
        EXPECT_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        A,
                        b,
                        startingPoint,
                        N,
                        shift,
                        model,
                        useParallelTempering)
        );
        EXPECT_TRUE(markovChain != nullptr);
    }
#endif //HOPS_MPI_SUPPORTED
}
