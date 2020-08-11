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
    std::ofstream createOutputStream(const std::string &outputPath, const std::string &description, int precision) {
        fs::path outPath(outputPath);
        outPath /= outPath.filename().string() + "_" + description + ".csv";
        std::ofstream out(outPath.string(), std::ios_base::app);
        out.precision(precision);
        return out;
    }
}

hops::CsvWriter::CsvWriter(std::string path, int outputPrecision) : path(std::move(path)),
                                                                    outputPrecision(outputPrecision) {
    fs::create_directories(CsvWriter::path);
}


void hops::CsvWriter::write(const std::string &description, const std::vector<float> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
    out.flush();
}

void hops::CsvWriter::write(const std::string &description, const std::vector<double> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
    out.flush();
}

void hops::CsvWriter::write(const std::string &description, const std::vector<long> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
    out.flush();
}

void hops::CsvWriter::write(const std::string &description, const std::vector<long double> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
    out.flush();
}

void hops::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
    out.flush();
}

void hops::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
    out.flush();
}

void hops::CsvWriter::write(const std::string &description,
                            const std::vector<Eigen::Matrix<long double, Eigen::Dynamic, 1>> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
    out.flush();
}


void hops::CsvWriter::write(const std::string &description, const std::vector<std::string> &records) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
    out.flush();
}

void hops::CsvWriter::write(const std::string &description, const Eigen::MatrixXd &matrix) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    for (long i = 0; i < matrix.rows(); ++i) {
        for (long j = 0; j < matrix.cols(); ++j) {
            out << matrix(i, j);
            if (j != matrix.cols() - 1) {
                out << ",";
            }
        }
        out << "\n";
    }
    out.flush();
}

void hops::CsvWriter::write(const std::string &description, const Eigen::VectorXd &vector) const {
    auto out = createOutputStream(CsvWriter::path, description, this->outputPrecision);
    for (long i = 0; i < vector.rows(); ++i) {
        out << vector(i);
        out << "\n";
    }
    out.flush();
}
