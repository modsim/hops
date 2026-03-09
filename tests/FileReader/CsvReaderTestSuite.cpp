#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CsvReaderTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

#include "hops/hops.hpp"

BOOST_AUTO_TEST_SUITE(CsvReader)

    BOOST_AUTO_TEST_CASE(readVectorOfDoubles) {
        Eigen::VectorXd expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXd>("../../resources/b_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfDoublesWithRowNames) {
        Eigen::VectorXd expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXd>("../../resources/b_small_with_row_names.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfDoubles) {
        Eigen::MatrixXd expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfDoublesWithColumnAndRowNames) {
        Eigen::MatrixXd expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        BOOST_CHECK(actualResult == expectedResult);
    }


    BOOST_AUTO_TEST_CASE(readSparseMatrixOfDoubles) {
        Eigen::MatrixXd matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<double> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small.csv");

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfDoublesWithColumnAndRowNames) {
        Eigen::MatrixXd matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<double> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
    }

BOOST_AUTO_TEST_SUITE_END()
