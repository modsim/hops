#define BOOST_TEST_MODULE Hdf5ReaderTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/FileReader/Hdf5Reader.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>


namespace {
    const char *hdf5TestFile = "../../resources/hdf5/test_data.hdf5";
}

BOOST_AUTO_TEST_SUITE(Hdf5Reader)

    BOOST_AUTO_TEST_CASE(readInt) {
        int expectedResult = 5;

        auto actualResult = hops::Hdf5Reader::read<int>(hdf5TestFile, "integer");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readFloat) {
        float expectedResult = 2.718;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "float");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readDouble) {
        double expectedResult = 3.141;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "double");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readString) {
        std::string expectedResult = "test";

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "string");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfInts) {
        Eigen::VectorXi expectedResult(4);
        expectedResult << 1, 2, 3, 4;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "vectorOfInts");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfFloats) {
        Eigen::VectorXf expectedResult(4);
        expectedResult << 1.1, 1.2, 1.3, 1.4;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "vectorOfFloats");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfDoubles) {
        Eigen::VectorXd expectedResult(4);
        expectedResult << 1.12, 1.23, 1.34, 1.45;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "vectorOfDoubles");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readVectorOfStrings) {
        std::vector<std::string> expectedResult{"test1", "test2", "test3"};

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "vectorOfStrings");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfInts) {
        Eigen::MatrixXi expectedResult(2, 2);
        expectedResult << 1, 2, 3, 4;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "matrixOfInts");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfFloats) {
        Eigen::MatrixXf expectedResult(2, 2);
        expectedResult << 1.1, 2.2, 3.3, 4.4;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "matrixOfFloats");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readMatrixOfDoubles) {
        Eigen::MatrixXd expectedResult(2, 2);
        expectedResult << 1.11, 2.22, 3.33, 4.44;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "matrixOfDoubles");

        BOOST_CHECK(actualResult == expectedResult);
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfInts) {
        Eigen::SparseMatrix<int> expectedResult(2, 2);
        expectedResult.insert(0, 0) = 1;
        expectedResult.insert(0, 1) = 2;
        expectedResult.insert(1, 0) = 3;
        expectedResult.insert(1, 1) = 4;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "matrixOfInts");

        for (long row = 0; row < actualResult.rows() || row < expectedResult.rows(); ++row) {
            for (long col = 0; col < actualResult.cols() || col < expectedResult.cols(); ++col) {
                BOOST_CHECK(actualResult.coeff(row, col) == expectedResult.coeff(row, col));
            }
        }
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfFloats) {
        Eigen::SparseMatrix<float> expectedResult(2, 2);
        expectedResult.insert(0, 0) = 1.1;
        expectedResult.insert(0, 1) = 2.2;
        expectedResult.insert(1, 0) = 3.3;
        expectedResult.insert(1, 1) = 4.4;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "matrixOfFloats");

        for (long row = 0; row < actualResult.rows() || row < expectedResult.rows(); ++row) {
            for (long col = 0; col < actualResult.cols() || col < expectedResult.cols(); ++col) {
                BOOST_CHECK(actualResult.coeff(row, col) == expectedResult.coeff(row, col));
            }
        }
    }

    BOOST_AUTO_TEST_CASE(readSparseMatrixOfDoubles) {
        Eigen::SparseMatrix<double> expectedResult(2, 2);
        expectedResult.insert(0, 0) = 1.11;
        expectedResult.insert(0, 1) = 2.22;
        expectedResult.insert(1, 0) = 3.33;
        expectedResult.insert(1, 1) = 4.44;

        auto actualResult = hops::Hdf5Reader::read<decltype(expectedResult)>(hdf5TestFile, "matrixOfDoubles");

        for (long row = 0; row < actualResult.rows() || row < expectedResult.rows(); ++row) {
            for (long col = 0; col < actualResult.cols() || col < expectedResult.cols(); ++col) {
                BOOST_CHECK(actualResult.coeff(row, col) == expectedResult.coeff(row, col));
            }
        }
    }

BOOST_AUTO_TEST_SUITE_END()

