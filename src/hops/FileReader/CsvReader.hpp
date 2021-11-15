#ifndef HOPS_CSVREADER_HPP
#define HOPS_CSVREADER_HPP

#include <fstream>
#include <string>
#include <vector>

namespace hops {
    class CsvReader {
    public:
        CsvReader() = delete;

        template<typename VectorType>
        static VectorType readVector(const std::string &file);

        template<typename MatrixType>
        static MatrixType readMatrix(const std::string &file, bool hasColumnAndRowNames = false);
    };
}

#endif //HOPS_CSVREADER_HPP
