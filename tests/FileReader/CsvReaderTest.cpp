#include <gtest/gtest.h>
#include <hops/FileReader/CsvReader.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace {
    TEST(CsvReader, readVectorOfInts) {
        Eigen::VectorXi expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXi>("../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readVectorOfIntsWithRowNames) {
        Eigen::VectorXi expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXi>("../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readVectorOfLongs) {
        Eigen::Matrix<long, Eigen::Dynamic, 1> expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::Matrix<long, Eigen::Dynamic, 1>>(
                "../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readVectorOfLongsWithRowNames) {
        Eigen::Matrix<long, Eigen::Dynamic, 1> expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::Matrix<long, Eigen::Dynamic, 1>>(
                "../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readVectorOfFloats) {
        Eigen::VectorXf expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXf>("../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readVectorOfFloatsWithRowNames) {
        Eigen::VectorXf expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXf>("../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readVectorOfDoubles) {
        Eigen::VectorXd expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXd>("../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readVectorOfDoublesWithRowNames) {
        Eigen::VectorXd expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXd>("../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfInts) {
        Eigen::MatrixXi expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXi>("../../resources/A_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfIntsWithColumnAndRowNames) {
        Eigen::MatrixXi expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXi>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfLongs) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>(
                "../../resources/A_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfLongsWithColumnAndRowNames) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>>(
                "../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfFloats) {
        Eigen::MatrixXf expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXf>("../../resources/A_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfFloatsWithColumnAndRowNames) {
        Eigen::MatrixXf expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXf>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfDoubles) {
        Eigen::MatrixXd expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readMatrixOfDoublesWithColumnAndRowNames) {
        Eigen::MatrixXd expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(CsvReader, readSparseMatrixOfInts) {
        Eigen::MatrixXi matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        Eigen::SparseMatrix<int> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<int>>("../../resources/A_small.csv");

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readSparseMatrixOfIntsWithColumnAndRowNames) {
        Eigen::MatrixXi matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        Eigen::SparseMatrix<int> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<int>>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readSparseMatrixOfLongs) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<long> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<long>>("../../resources/A_small.csv");

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readSparseMatrixOfLongsWithColumnAndRowNames) {
        Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> matrix(5, 4);
        matrix << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<long> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<long>>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readSparseMatrixOfFloats) {
        Eigen::MatrixXf matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<float> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<float>>("../../resources/A_small.csv");

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readSparseMatrixOfFloatsWithColumnAndRowNames) {
        Eigen::MatrixXf matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<float> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::SparseMatrix<float>>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readSparseMatrixOfDoubles) {
        Eigen::MatrixXd matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<double> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small.csv");

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readSparseMatrixOfDoublesWithColumnAndRowNames) {
        Eigen::MatrixXd matrix(5, 4);
        matrix << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::SparseMatrix<double> expectedResult = matrix.sparseView();

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_TRUE((actualResult - expectedResult).norm() <= 0);
    }

    TEST(CsvReader, readRecon2v04) {
        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/Recon2.v04/A_Recon2.v04_unrounded.csv");

        for (long i = 0; i < actualResult.cols(); ++i) {
            EXPECT_TRUE((actualResult.col(i).array() != 0).any());
        }
    }
}
