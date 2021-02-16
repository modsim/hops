#ifndef HOPS_CSVWRITER_HPP
#define HOPS_CSVWRITER_HPP

#include "FileWriter.hpp"

namespace hops {
    class CsvWriter : public FileWriter {
    public:
        explicit CsvWriter(std::string path, int outputPrecision = 17);

        void write(const std::string &description, const std::vector<float> &records) const override;

        void write(const std::string &description, const std::vector<double> &records) const override;

        void write(const std::string &description, const std::vector<long> &records) const override;

        void write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const override;

        void write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const override;

        void write(const std::string &description, const std::vector<std::string> &records) const override;

        void write(const std::string &description, const Eigen::MatrixXd &matrix) const override;

        void write(const std::string &description, const Eigen::VectorXd &vector) const override;

    private:
        std::string path;
        int outputPrecision;
    };
}

#endif //HOPS_CSVWRITER_HPP
