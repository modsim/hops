#include <Eigen/Core>
#include <Eigen/Sparse>

#include "CsvReader.hpp"

template<typename VectorType>
VectorType hops::CsvReader::readVector(const std::string &file) {
    std::ifstream fileStream(file);
    if (fileStream.fail()) {
        throw std::runtime_error("Could not access file: " + file);
    }
    std::vector<std::string> cells;
    std::string line;
    while (std::getline(fileStream, line)) {
        auto const position = line.find_last_of(',');
        std::string cell = line.substr(position + 1);
        if (!cell.empty()) {
            cells.push_back(cell);
        }
    }

    size_t startIndex = 0;
    try {
        std::stod(cells[0]);
    }
    catch (std::invalid_argument &) {
        startIndex = 1;
    }
    VectorType result(cells.size() - startIndex);
    for (size_t i = startIndex; i < cells.size(); ++i) {
        result(i - startIndex) = std::stod(cells[i]);
    }
    return result;
}

template Eigen::VectorXi hops::CsvReader::readVector(const std::string &file);

template Eigen::Matrix<long, Eigen::Dynamic, 1> hops::CsvReader::readVector(const std::string &file);

template Eigen::VectorXf hops::CsvReader::readVector(const std::string &file);

template Eigen::VectorXd hops::CsvReader::readVector(const std::string &file);


template<typename MatrixType>
MatrixType hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames) {

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
            cells.emplace_back(cell);
        }
        stringRepresentationOfMatrix.emplace_back(cells);
    }

    size_t numberOfRows = hasColumnAndRowNames ? stringRepresentationOfMatrix.size() - 1
                                               : stringRepresentationOfMatrix.size();
    size_t numberOfColumns = hasColumnAndRowNames ? stringRepresentationOfMatrix.at(0).size() - 1
                                                  : stringRepresentationOfMatrix.at(0).size();
    Eigen::SparseMatrix<typename MatrixType::Scalar> result(numberOfRows, numberOfColumns);
    std::vector<Eigen::Triplet<typename MatrixType::Scalar>> triplets;

    for (size_t i = 0; i < numberOfRows; ++i) {
        for (size_t j = 0; j < numberOfColumns; ++j) {
            size_t row = hasColumnAndRowNames ? i + 1 : i;
            size_t col = hasColumnAndRowNames ? j + 1 : j;
            typename MatrixType::Scalar value = std::stod(stringRepresentationOfMatrix.at(row).at(col));
            if (value != 0) {
                triplets.emplace_back(i, j, value);
            }
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

template Eigen::MatrixXi hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);

template Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>
hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);

template Eigen::MatrixXf hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);

template Eigen::MatrixXd hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);

template Eigen::SparseMatrix<int> hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);

template Eigen::SparseMatrix<long> hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);

template Eigen::SparseMatrix<float> hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);

template Eigen::SparseMatrix<double> hops::CsvReader::readMatrix(const std::string &file, bool hasColumnAndRowNames);
