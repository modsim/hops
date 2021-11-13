#define BOOST_TEST_MODULE MarkovChainFactoryTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/hops.hpp>

namespace {
    class ModelMock : public hops::Model {
    public:
        [[nodiscard]] double computeNegativeLogLikelihood(const hops::VectorType &x) const override {
            return 0;
        }

        [[nodiscard]] std::optional<hops::VectorType>
        computeLogLikelihoodGradient(const hops::VectorType &x) const override {
            return Eigen::VectorXd::Ones(2);
        }

        [[nodiscard]] std::optional<hops::MatrixType>
        computeExpectedFisherInformation(const hops::VectorType &type) const override {
            return Eigen::MatrixXd::Identity(2, 2);
        }

        std::unique_ptr<Model> deepCopy() const override {
            return std::make_unique<ModelMock>();
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
        model = std::make_shared<ModelMock>();
    }

    Eigen::MatrixXd A, N;
    Eigen::VectorXd b, startingPoint, shift;
    std::shared_ptr<hops::Model> model;
};

BOOST_AUTO_TEST_SUITE(MarkovchainFactory)

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