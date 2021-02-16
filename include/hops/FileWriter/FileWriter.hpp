#ifndef HOPS_FILEWRITER_HPP
#define HOPS_FILEWRITER_HPP

#include <Eigen/Core>
#include <string>
#include <vector>

namespace hops {
    class FileWriter {
    public:
        virtual ~FileWriter() = default;

        /**
         * @brief Writes records.
         * @param description Description to attach to the records.
         * @param records
         */
        virtual void write(const std::string &description, const std::vector<float> &records) const = 0;

        /**
         * @brief Writes records.
         * @param description Description to attach to the records.
         * @param records
         */
        virtual void write(const std::string &description, const std::vector<double> &records) const = 0;

        /**
         * @brief Writes records.
         * @param description Description to attach to the records.
         * @param records
         */
        virtual void write(const std::string &description, const std::vector<long> &records) const = 0;

        /**
         * @brief Writes records.
         * @param description Description to attach to the records.
         * @param records
         */
        virtual void write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const = 0;

        /**
         * @brief Writes records.
         * @param description Description to attach to the records.
         * @param records
         */
        virtual void write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const = 0;

        /**
         * @brief Writes records.
         * @param description Description to attach to the records.
         * @param records
         */
        virtual void write(const std::string &description, const std::vector<std::string> &records) const = 0;

        /**
         * @brief Writes a single matrix in double precision.
         * @param description
         * @param matrix
         */
        virtual void write(const std::string &description, const Eigen::MatrixXd &matrix) const = 0;

        /**
         * @brief Writes a single vector in double precision.
         * @param description
         * @param matrix
         */
        virtual void write(const std::string &description, const Eigen::VectorXd &vector) const = 0;

    };
}

#endif //HOPS_FILEWRITER_HPP
