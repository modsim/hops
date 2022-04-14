#define BOOST_TEST_MODULE hitAndRunProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>

BOOST_AUTO_TEST_SUITE(HitAndRunProposal)

    BOOST_AUTO_TEST_CASE(Cube) {
        const long rows = 6;
        const long cols = 3;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                -1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        hops::HitAndRunProposal hitAndRunProposal(A, b, interiorPoint);
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = hitAndRunProposal.propose(randomNumberGenerator);
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            hitAndRunProposal.acceptProposal();
        }
    }

    BOOST_AUTO_TEST_CASE(CubeStartOnBorder) {
        const long rows = 6;
        const long cols = 3;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                -1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 1;
        }

        hops::HitAndRunProposal hitAndRunProposal(A, b, interiorPoint);
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = hitAndRunProposal.propose(randomNumberGenerator);
            BOOST_CHECK(((b - A * proposal).array() >= 0).all());
            hitAndRunProposal.acceptProposal();
        }
    }


    BOOST_AUTO_TEST_CASE(ExceptionIsThrownForOpenCubeAndUniformChord) {
        const long rows = 3;
        const long cols = 3;
        Eigen::MatrixXd A(rows, cols);
        A << 0, 0, 0,
                0, 0, 0,
                0, 0, 0;

        Eigen::VectorXd b(rows);
        b << 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        hops::HitAndRunProposal<decltype(A), decltype(b), hops::UniformStepDistribution<double>>
                hitAndRunProposal(A, b, interiorPoint);

        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            BOOST_CHECK_THROW(hitAndRunProposal.propose(randomNumberGenerator),
                              std::invalid_argument
            );
        }
    }

    BOOST_AUTO_TEST_CASE(OpenCubeAndGaussianChord) {
        const long rows = 3;
        const long cols = 3;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0,
                0, 1, 0,
                0, 0, 1;

        Eigen::VectorXd b(rows);
        b << 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        hops::HitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>>
                hitAndRunProposal(A, b, interiorPoint);

        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = hitAndRunProposal.propose(randomNumberGenerator);
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            hitAndRunProposal.acceptProposal();
        }
    }

    BOOST_AUTO_TEST_CASE(CubeAndGaussianChord) {
        const long rows = 6;
        const long cols = 3;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                -1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        hops::HitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>>
                hitAndRunProposal(A, b, interiorPoint);
        hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = hitAndRunProposal.propose(randomNumberGenerator);
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            hitAndRunProposal.computeLogAcceptanceProbability();
            hitAndRunProposal.acceptProposal();
        }
    }


BOOST_AUTO_TEST_SUITE_END()


