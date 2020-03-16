#include <gtest/gtest.h>
#include <hops/FileReader/CsvReader.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace {
    TEST(FileReader, readVectorOfInts) {
        Eigen::VectorXi expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXi>("../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readVectorOfIntsWithRowNames) {
        Eigen::VectorXi expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXi>("../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readVectorOfLongs) {
        Eigen::Matrix<long, Eigen::Dynamic, 1> expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::Matrix<long, Eigen::Dynamic, 1>>(
                "../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readVectorOfLongsWithRowNames) {
        Eigen::Matrix<long, Eigen::Dynamic, 1> expectedResult(5);
        expectedResult << 1, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::Matrix<long, Eigen::Dynamic, 1>>(
                "../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readVectorOfFloats) {
        Eigen::VectorXf expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXf>("../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readVectorOfFloatsWithRowNames) {
        Eigen::VectorXf expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXf>("../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readVectorOfDoubles) {
        Eigen::VectorXd expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXd>("../../resources/b_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readVectorOfDoublesWithRowNames) {
        Eigen::VectorXd expectedResult(5);
        expectedResult << 1.5, 0, -1, 0, 0;

        auto actualResult = hops::CsvReader::readVector<Eigen::VectorXd>("../../resources/b_small_with_row_names.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readMatrixOfInts) {
        Eigen::MatrixXi expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXi>("../../resources/A_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readMatrixOfIntsWithColumnAndRowNames) {
        Eigen::MatrixXi expectedResult(5, 4);
        expectedResult << 1, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXi>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readMatrixOfLongs) {
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

    TEST(FileReader, readMatrixOfLongsWithColumnAndRowNames) {
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

    TEST(FileReader, readMatrixOfFloats) {
        Eigen::MatrixXf expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXf>("../../resources/A_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readMatrixOfFloatsWithColumnAndRowNames) {
        Eigen::MatrixXf expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXf>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readMatrixOfDoubles) {
        Eigen::MatrixXd expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small.csv");

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readMatrixOfDoublesWithColumnAndRowNames) {
        Eigen::MatrixXd expectedResult(5, 4);
        expectedResult << 1.5, 1, 1, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;

        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../resources/A_small_with_column_and_row_names.csv", true);

        EXPECT_EQ(actualResult, expectedResult);
    }

    TEST(FileReader, readSparseMatrixOfInts) {
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

    TEST(FileReader, readSparseMatrixOfIntsWithColumnAndRowNames) {
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

    TEST(FileReader, readSparseMatrixOfLongs) {
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

    TEST(FileReader, readSparseMatrixOfLongsWithColumnAndRowNames) {
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

    TEST(FileReader, readSparseMatrixOfFloats) {
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

    TEST(FileReader, readSparseMatrixOfFloatsWithColumnAndRowNames) {
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

    TEST(FileReader, readSparseMatrixOfDoubles) {
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

    TEST(FileReader, readSparseMatrixOfDoublesWithColumnAndRowNames) {
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

    TEST(FileReader, readRecon2v04) {
        auto actualResult = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/Recon2.v04/A_Recon2.v04_unrounded.csv");

        for (long i = 0; i < actualResult.cols(); ++i) {
            EXPECT_TRUE((actualResult.col(i).array() != 0).any());
        }
    }
}
