#ifndef HOPS_CSVWRITER_HPP
#define HOPS_CSVWRITER_HPP

#include "FileWriter.hpp"

namespace hops {
    class CsvWriter : public FileWriter {
    public:
        explicit CsvWriter(std::string path);

        void write(const std::string &description, const std::vector<float> &records) const override;

        void write(const std::string &description, const std::vector<double> &records) const override;

        void write(const std::string &description, const std::vector<long> &records) const override;

        void write(const std::string &description, const std::vector<long double> &records) const override;

        void write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const override;

        void write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const override;

        void write(const std::string &description, const std::vector<std::string> &records) const override;

        void write(const std::string &description,
                   const std::vector<Eigen::Matrix<long double, Eigen::Dynamic, 1>> &records) const override;

        // TODO add to interface
        void write(const std::string &description, const Eigen::MatrixXd &vector);

        // TODO add to interface
        void write(const std::string &description, const Eigen::VectorXd &vector);

    private:
        std::string path;
    };
}

#endif //HOPS_CSVWRITER_HPP
