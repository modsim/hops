#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CsvWriteImplTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include "hops/hops.hpp"

BOOST_AUTO_TEST_SUITE(CsvWriterImpl)

    BOOST_AUTO_TEST_CASE( writeInts) {
        std::vector<int> data{-525, 0, 35};
        std::string expectedOutput = "-525\n0\n35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        BOOST_CHECK(actualOutput.str() == expectedOutput);
    }

    BOOST_AUTO_TEST_CASE( writeLongs) {
        std::vector<long> data{-525, 0, 35};
        std::string expectedOutput = "-525\n0\n35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        BOOST_CHECK(actualOutput.str() == expectedOutput);
    }

    BOOST_AUTO_TEST_CASE( writeFloats) {
        std::vector<float> data{1.25f, 5.513f, 8.35f};
        std::string expectedOutput = "1.25\n5.513\n8.35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        BOOST_CHECK(actualOutput.str() == expectedOutput);
    }

    BOOST_AUTO_TEST_CASE( writeDoubles) {
        std::vector<double> data{1.25, 5.513, 8.35};
        std::string expectedOutput = "1.25\n5.513\n8.35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        BOOST_CHECK(actualOutput.str() == expectedOutput);
    }

    BOOST_AUTO_TEST_CASE( writeStrings) {
        std::vector<std::string> data{"test1", "test2"};
        std::string expectedOutput = "test1\ntest2\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        BOOST_CHECK(actualOutput.str() == expectedOutput);
    }

    BOOST_AUTO_TEST_CASE( writeEigenFloatVectors) {
        std::vector<Eigen::VectorXf> data;
        Eigen::VectorXf v1(3);
        v1 << 1.1f,5.6f,3.14f;
        data.emplace_back(v1);
        Eigen::VectorXf v2(1);
        v2 << -1.1f;
        data.emplace_back(v2);
        std::string expectedOutput = "1.1,5.6,3.14\n-1.1\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeEigenVectorRecords(actualOutput, data);

        BOOST_CHECK(actualOutput.str() == expectedOutput);
    }

    BOOST_AUTO_TEST_CASE( writeEigenDoubleVectors) {
        std::vector<Eigen::VectorXd> data;
        Eigen::VectorXd v1(3);
        v1 << 1.1,5.6,3.14;
        data.emplace_back(v1);
        Eigen::VectorXd v2(1);
        v2 << -1.1;
        data.emplace_back(v2);
        std::string expectedOutput = "1.1,5.6,3.14\n-1.1\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeEigenVectorRecords(actualOutput, data);

        BOOST_CHECK(actualOutput.str() == expectedOutput);
    }

BOOST_AUTO_TEST_SUITE_END()
