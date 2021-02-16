#define BOOST_TEST_MODULE MarkovChainFactoryTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/MarkovChain/MarkovChainFactory.hpp>

namespace {
    class ModelMock {
    public:
        using VectorType = Eigen::VectorXd;

        [[maybe_unused]] static double calculateNegativeLogLikelihood(const VectorType &) {
            return 0.;
        }

        [[maybe_unused]] static Eigen::VectorXd calculateLogLikelihoodGradient(const VectorType &) {
            return Eigen::VectorXd::Ones(2);
        }

        [[maybe_unused]] static Eigen::MatrixXd calculateExpectedFisherInformation(const VectorType &) {
            return Eigen::MatrixXd::Identity(2, 2);
        }
    };
}

struct MarkovChainFactoryTestFixture {
public:
    MarkovChainFactoryTestFixture() {
        A = Eigen::MatrixXd(4, 2);
        A << Eigen::MatrixXd::Identity(2, 2), -Eigen::MatrixXd::Identity(2, 2);
        b = Eigen::VectorXd::Ones(4);
        startingPoint = 0.5 * Eigen::VectorXd::Ones(2);
        N = Eigen::MatrixXd::Identity(2, 2);
        shift = Eigen::VectorXd::Ones(2);
    }

    Eigen::MatrixXd A, N;
    Eigen::VectorXd b, startingPoint, shift;
    ModelMock model;
};

BOOST_AUTO_TEST_SUITE(MarkovchainFactory)

    BOOST_AUTO_TEST_CASE(createUniformCoordinateHitAndRun) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        auto fixture = MarkovChainFactoryTestFixture();
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createUniformDikinWalk) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        auto fixture = MarkovChainFactoryTestFixture();
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createUniformHitAndRun) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        auto fixture = MarkovChainFactoryTestFixture();
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createUniformCoordinateHitAndRunRounded) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        auto fixture = MarkovChainFactoryTestFixture();
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createUniformDikinWalkRounded) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        auto fixture = MarkovChainFactoryTestFixture();
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createUniformHitAndRunRounded) {
        std::unique_ptr<hops::MarkovChain> markovChain;
        auto fixture = MarkovChainFactoryTestFixture();
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformCoordinateHitAndRun) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalk) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRun) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformCoordinateHitAndRunRounded) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalkRounded) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRunRounded) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

#ifdef HOPS_MPI_SUPPORTED

    BOOST_AUTO_TEST_CASE(createNonUniformCoordinateHitAndRunWithParallelTempering) {
        bool useParallelTempering = true;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalkWithParallelTempering) {
        bool useParallelTempering = true;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRunWithParallelTempering) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformCoordinateHitAndRunRoundedWithParallelTempering) {
        bool useParallelTempering = true;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalkRoundedWithParallelTempering) {
        bool useParallelTempering = true;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRunRoundedWithParallelTempering) {
        bool useParallelTempering = false;
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        useParallelTempering)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

#endif //HOPS_MPI_SUPPORTED

BOOST_AUTO_TEST_SUITE_END()