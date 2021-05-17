#ifndef HOPS_CHAINDATA_HPP
#define HOPS_CHAINDATA_HPP

#include "../FileWriter/FileWriter.hpp"
#include "Exceptions.hpp"

#include <Eigen/Core>

#include <vector>
#include <memory>

namespace hops {
    class ChainData {
    public:
        ChainData() : acceptanceRates(nullptr),
                      negativeLogLikelihood(nullptr),
                      states(nullptr),
                      timestamps(nullptr) {}


        [[nodiscard]] const std::vector<double> &getAcceptanceRates() const {
            if (!acceptanceRates) throw UninitializedDataFieldException("acceptanceRates");
            return *acceptanceRates;
        }

        void setAcceptanceRates(const std::shared_ptr<std::vector<double>> &acceptanceRates) {
            this->acceptanceRates = std::shared_ptr<std::vector<double>>(acceptanceRates);
        }


        [[nodiscard]] const std::vector<double> &getNegativeLogLikelihood() const {
            if (!negativeLogLikelihood) throw UninitializedDataFieldException("negativeLogLikelihood");
            return *negativeLogLikelihood;
        }

        void setNegativeLogLikelihood(const std::shared_ptr<std::vector<double>> &negativeLogLikelihood) {
            this->negativeLogLikelihood = std::shared_ptr<std::vector<double>>(negativeLogLikelihood);
        }


        [[nodiscard]] const std::vector<Eigen::VectorXd> &getStates() const {
            if (!states) throw UninitializedDataFieldException("states");
            return *states;
        }

        void setStates(const std::shared_ptr<std::vector<Eigen::VectorXd>> &states) {
            this->states = std::shared_ptr<std::vector<Eigen::VectorXd>>(states);
        }


        [[nodiscard]] const std::vector<long> &getTimestamps() const {
            if (!timestamps) throw UninitializedDataFieldException("timestamps");
            return *timestamps;
        }

        void setTimestamps(const std::shared_ptr<std::vector<long>> timestamps) {
            this->timestamps = std::shared_ptr<std::vector<long>>(timestamps);
        }


        void write(FileWriter *const fileWriter) const {
            if (acceptanceRates) {
                fileWriter->write("acceptanceRates", *acceptanceRates);
            }

            if (negativeLogLikelihood) {
                fileWriter->write("negativeLogLikelihood", *negativeLogLikelihood);
            }

            if (states) {
                fileWriter->write("states", *states);
            }

            if (timestamps) {
                fileWriter->write("timestamps", *timestamps);
            }
        }

        void reset() {
            acceptanceRates->clear();
            negativeLogLikelihood->clear();
            states->clear();
            timestamps->clear();
        }

    private:
        std::shared_ptr<std::vector<double>> acceptanceRates;
        std::shared_ptr<std::vector<double>> negativeLogLikelihood;
        std::shared_ptr<std::vector<Eigen::VectorXd>> states;
        std::shared_ptr<std::vector<long>> timestamps;

        friend class Data;
    };
}

#endif // HOPS_CHAINDATA_HPP

