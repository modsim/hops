#define BOOST_TEST_MODULE CsvReaderTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <hops/hops.hpp>

BOOST_AUTO_TEST_SUITE(CsvReader)

    BOOST_AUTO_TEST_CASE(readVectorOfInts) {
        Eigen::VectorXi expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXi>("../../resources/b_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfIntsWithRowNames) {
        Eigen::VectorXi expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXi>("../../resources/b_small_with_row_names.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfLongs) {
        Eigen::Matrix<long, Eigen::Dynamic, 1> expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::Matrix<long, Eigen::Dynamic, 1>>(
                "../../resources/b_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfLongsWithRowNames) {
        Eigen::Matrix<long, Eigen::Dynamic, 1> expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::Matrix<long, Eigen::Dynamic, 1>>(
                "../../resources/b_small_with_row_names.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfFloats) {
        Eigen::VectorXf expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXf>("../../resources/b_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfFloatsWithRowNames) {
        Eigen::VectorXf expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXf>("../../resources/b_small_with_row_names.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

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

    BOOST_AUTO_TEST_CASE(readMatrixOfInts) {
        Eigen::MatrixXi expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXi>("../../resources/A_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfIntsWithColumnAndRowNames) {
        Eigen::MatrixXi expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXi>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfLongs) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>(
                "../../resources/A_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfLongsWithColumnAndRowNames) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfFloats) {
        Eigen::MatrixXf expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXf>("../../resources/A_small.csv");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfFloatsWithColumnAndRowNames) {
        Eigen::MatrixXf expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXf>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

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

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfInts) {
        Eigen::MatrixXi matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        Eigen::SparseMatrix<int> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<int>>("../../resources/A_small.csv");

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfIntsWithColumnAndRowNames) {
        Eigen::MatrixXi matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        Eigen::SparseMatrix<int> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<int>>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfLongs) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<long> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<long>>("../../resources/A_small.csv");

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfLongsWithColumnAndRowNames) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<long> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<long>>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfFloats) {
        Eigen::MatrixXf matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<float> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<float>>("../../resources/A_small.csv");

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfFloatsWithColumnAndRowNames) {
        Eigen::MatrixXf matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<float> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<float>>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        BOOST_CHECK((actualResult - expectedResult).norm() <= 0);
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
