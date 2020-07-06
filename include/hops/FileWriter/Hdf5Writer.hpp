#ifndef HOPS_HDF5WRITER_HPP
#define HOPS_HDF5WRITER_HPP

#include "FileWriter.hpp"

namespace hops {
    /**
     * @brief Beware of strange behaviours that originate from underlying HDF5 C-API
     * @details 1) Don't close file, if you want to append more data.
     *          2) For strings, data can not be appended.
     */
    class Hdf5Writer : public FileWriter {
    public:
        explicit Hdf5Writer(std::string path);
        ~Hdf5Writer() override;

        void write(const std::string &description, const std::vector<float> &records) const override;

        void write(const std::string &description, const std::vector<double> &records) const override;

        void write(const std::string &description, const std::vector<long> &records) const override;

        void write(const std::string &description, const std::vector<long double> &records) const override;

        void write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const override;

        void write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const override;

        void write(const std::string &description,
                   const std::vector<Eigen::Matrix<long double, Eigen::Dynamic, 1>> &records) const override;

        void write(const std::string &description, const std::vector<std::string> &records) const override;

        void write(const std::string &description, const Eigen::MatrixXd &matrix) const override;

        void write(const std::string &description, const Eigen::VectorXd &vector) const override;

    private:
        std::string path;
        int64_t fileId = -1;
    };
}

#endif //HOPS_HDF5WRITER_HPP
