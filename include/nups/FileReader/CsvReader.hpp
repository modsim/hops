#ifndef NUPS_CSVREADER_HPP
#define NUPS_CSVREADER_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <fstream>
#include <string>
#include <vector>

namespace nups {
    class CsvReader {
    public:
        template<typename VectorType>
        static VectorType readVector(const std::string &file);

        template<typename MatrixType>
        static MatrixType readMatrix(const std::string &file);
    };
}

#endif //NUPS_CSVREADER_HPP
