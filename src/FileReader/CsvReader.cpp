#include "hops/FileReader/CsvReader.hpp"

template<typename VectorType>
VectorType hops::CsvReader::readVector(const std::string &file) {
    std::ifstream fileStream(file);
    if (fileStream.fail()) {
        throw std::runtime_error("Could not access file: " + file);
    }
    std::vector<std::string> cells;
    std::string line;
    while (std::getline(fileStream, line)) {
        cells.push_back(line);
    }

    VectorType result(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        result(i) = std::stod(cells[i]);
    }
    return result;
}

template Eigen::VectorXi hops::CsvReader::readVector(const std::string &file);

template Eigen::Matrix<long, Eigen::Dynamic, 1> hops::CsvReader::readVector(const std::string &file);

template Eigen::VectorXf hops::CsvReader::readVector(const std::string &file);

template Eigen::VectorXd hops::CsvReader::readVector(const std::string &file);


template<typename MatrixType>
MatrixType hops::CsvReader::readMatrix(const std::string &file) {

    std::ifstream fileStream(file);
    if (fileStream.fail()) {
        throw std::runtime_error("Could not access file: " + file);
    }
    std::string line;
    std::vector<std::vector<std::string>> stringRepresentationOfMatrix;
    while (std::getline(fileStream, line)) {
        std::stringstream lineStream(line);
        std::vector<std::string> cells;
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            cells.push_back(cell);
        }
        stringRepresentationOfMatrix.push_back(cells);
    }

    Eigen::SparseMatrix<typename MatrixType::Scalar> result(stringRepresentationOfMatrix.size(),
                                                            stringRepresentationOfMatrix.at(0).size());
    std::vector<Eigen::Triplet<typename MatrixType::Scalar>> triplets;

    for (size_t i = 0; i < stringRepresentationOfMatrix.size(); ++i) {
        for (size_t j = 0; j < stringRepresentationOfMatrix.at(i).size(); ++j) {
            typename MatrixType::Scalar value = std::stod(stringRepresentationOfMatrix.at(i).at(j));
            if (value != 0) {
                triplets.emplace_back(i, j, value);
            }
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

template Eigen::MatrixXi hops::CsvReader::readMatrix(const std::string &file);

template Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> hops::CsvReader::readMatrix(const std::string &file);

template Eigen::MatrixXf hops::CsvReader::readMatrix(const std::string &file);

template Eigen::MatrixXd hops::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<int> hops::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<long> hops::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<float> hops::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<double> hops::CsvReader::readMatrix(const std::string &file);
