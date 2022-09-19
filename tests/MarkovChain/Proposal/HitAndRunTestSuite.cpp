#define BOOST_TEST_MODULE hitAndRunProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>

#include "hops/MarkovChain/Proposal/HitAndRunProposal.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

BOOST_AUTO_TEST_SUITE(HitAndRunProposal)

    BOOST_AUTO_TEST_CASE(DimensionNames) {
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

        std::vector<std::string> expectedNames = {"x_0", "x_1", "x_2"};
        auto actualNames = hitAndRunProposal.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for (size_t i = 0; i < expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }

        expectedNames = std::vector<std::string>{"y_1", "y_2", "y_3"};
        hitAndRunProposal.setDimensionNames(expectedNames);
        actualNames = hitAndRunProposal.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for (size_t i = 0; i < expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }
    }

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
            double acceptancePropability = hitAndRunProposal.computeLogAcceptanceProbability();
            BOOST_CHECK(std::isfinite(acceptancePropability));
            hitAndRunProposal.acceptProposal();
        }
    }


BOOST_AUTO_TEST_SUITE_END()


