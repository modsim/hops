#include <Eigen/Core>
#include <nups/FileWriter/CsvWriterImpl.hpp>
#include <iostream>

template<typename Derived>
void nups::internal::CsvWriterImpl::writeOneDimensionalRecords(std::ostream &out, const std::vector<Derived> &records) {
    for (const auto &record: records) {
        out << record << "\n";
    }
}

template void
nups::internal::CsvWriterImpl::writeOneDimensionalRecords(std::ostream &out, const std::vector<int> &records);

template void
nups::internal::CsvWriterImpl::writeOneDimensionalRecords(std::ostream &out, const std::vector<long> &records);

template void
nups::internal::CsvWriterImpl::writeOneDimensionalRecords(std::ostream &out, const std::vector<float> &records);

template void
nups::internal::CsvWriterImpl::writeOneDimensionalRecords(std::ostream &out, const std::vector<double> &records);

template void
nups::internal::CsvWriterImpl::writeOneDimensionalRecords(std::ostream &out, const std::vector<std::string> &records);


template<typename Derived>
void nups::internal::CsvWriterImpl::writeEigenVectorRecords(std::ostream &out, const std::vector<Derived> &records) {
    for (const auto &record: records) {
        for (long i = 0; i < record.rows() - 1; ++i) {
            out << record(i) << ",";
        }
        out << record(record.rows() - 1) << "\n";
    }
}

template void
nups::internal::CsvWriterImpl::writeEigenVectorRecords(std::ostream &out, const std::vector<Eigen::VectorXf> &records);

template void
nups::internal::CsvWriterImpl::writeEigenVectorRecords(std::ostream &out, const std::vector<Eigen::VectorXd> &records);
