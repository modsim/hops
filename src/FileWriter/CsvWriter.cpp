#include <fstream>
#include <hops/FileWriter/CsvWriter.hpp>
#include "hops/FileWriter/CsvWriterImpl.hpp"

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

namespace {
    // TODO replace hard-coded precision by user choice
    std::ofstream createOutputStream(std::string outputPath, std::string description) {
        fs::path outPath(outputPath);
        outPath /= outPath.filename().string() + "_" + description + ".csv";
        std::ofstream out(outPath.string(), std::ios_base::app);
        out.precision(17);
        return out;
    }
}

hops::CsvWriter::CsvWriter(std::string path) : path(std::move(path)) {
    fs::create_directories(CsvWriter::path);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<float> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<double> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<long> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<long double> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description,
                            const std::vector<Eigen::Matrix<long double, Eigen::Dynamic, 1>> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
}


void hops::CsvWriter::write(const std::string &description, const std::vector<std::string> &records) const {
    auto out = createOutputStream(CsvWriter::path, description);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const Eigen::MatrixXd &matrix) {
    auto out = createOutputStream(CsvWriter::path, description);
    // TODO move implementaiton to writerImpl
    for (long i = 0; i < matrix.rows(); ++i) {
        for (long j = 0; j < matrix.cols(); ++j) {
            out << matrix(i, j);
            if (j != matrix.cols() - 1) {
                out << ",";
            }
        }
        out << "\n";
    }
}

void hops::CsvWriter::write(const std::string &description, const Eigen::VectorXd &vector) {
    auto out = createOutputStream(CsvWriter::path, description);
    // TODO move implementaiton to writerImpl
    for (long i = 0; i < vector.rows(); ++i) {
        out << vector(i);
        out << "\n";
    }
}
