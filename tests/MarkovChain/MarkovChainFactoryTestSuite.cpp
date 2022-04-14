#define BOOST_TEST_MODULE MarkovChainFactoryTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/hops.hpp>
#include <Eigen/Dense>

namespace {
    class ModelMock {
    public:
        [[maybe_unused]] double computeNegativeLogLikelihood(const hops::VectorType &) {
            return 0.;
        }

        std::optional<hops::VectorType> computeLogLikelihoodGradient(const hops::VectorType &x) {
            return std::nullopt;
        }

        std::optional<hops::MatrixType> computeExpectedFisherInformation(const hops::VectorType &type) {
            return std::nullopt;
        }

        bool hasConstantExpectedFisherInformation() {
            return false;
        }

        std::vector<std::string> getDimensionNames() const {
            return {};
        }

        std::unique_ptr<hops::Model> copyModel() const {
            return nullptr;
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
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalk) {
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRun) {
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformCoordinateHitAndRunRounded) {
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
                        fixture.model)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalkRounded) {
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChain(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model),
                std::runtime_error
        );
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRunRounded) {
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
                        fixture.model)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

#ifdef HOPS_MPI_SUPPORTED

    BOOST_AUTO_TEST_CASE(createNonUniformCoordinateHitAndRunWithParallelTempering) {
        hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        synchronizedRandomNumberGenerator)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalkWithParallelTempering) {
        hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        synchronizedRandomNumberGenerator)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRunWithParallelTempering) {
        hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.model,
                        synchronizedRandomNumberGenerator)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformCoordinateHitAndRunRoundedWithParallelTempering) {
        hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
                        hops::MarkovChainType::CoordinateHitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        synchronizedRandomNumberGenerator)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

    BOOST_AUTO_TEST_CASE(createNonUniformDikinWalkRoundedWithParallelTempering) {
        hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
                        hops::MarkovChainType::DikinWalk,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        synchronizedRandomNumberGenerator),
                std::runtime_error
        );
    }

    BOOST_AUTO_TEST_CASE(createNonUniformHitAndRunRoundedWithParallelTempering) {
        hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);
        auto fixture = MarkovChainFactoryTestFixture();
        std::unique_ptr<hops::MarkovChain> markovChain;
        BOOST_CHECK_NO_THROW(
                markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
                        hops::MarkovChainType::HitAndRun,
                        fixture.A,
                        fixture.b,
                        fixture.startingPoint,
                        fixture.N,
                        fixture.shift,
                        fixture.model,
                        synchronizedRandomNumberGenerator)
        );
        BOOST_CHECK(markovChain != nullptr);
    }

#endif //HOPS_MPI_SUPPORTED

BOOST_AUTO_TEST_SUITE_END()