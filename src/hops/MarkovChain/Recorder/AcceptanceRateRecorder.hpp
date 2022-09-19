#ifndef HOPS_ACCEPTANCERATERECORDER_HPP
#define HOPS_ACCEPTANCERATERECORDER_HPP

#include <memory>
#include <vector>

#include "hops/FileWriter/FileWriter.hpp"
#include "hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp"
#include "hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp"
#include "hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp"

namespace hops {
    template<typename MarkovChainImpl>
    class AcceptanceRateRecorder : public MarkovChainImpl {
    public:
        explicit AcceptanceRateRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {
            records = std::make_shared<std::vector<double>>();
        }

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("acceptance rates", *records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        };

        [[nodiscard]] std::vector<double> getAcceptanceRateRecords() const {
            return *records;
        }

        void storeRecord() {
            records->emplace_back(MarkovChainImpl::getAcceptanceRate());
            if constexpr(IsStoreRecordAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::storeRecord();
            }
        }

        void clearRecords() {
            records->clear();
            if constexpr(IsClearRecordsAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::clearRecords();
            }
        }

    private:
        std::shared_ptr<std::vector<double>> records;
    };
}

#endif //HOPS_ACCEPTANCERATERECORDER_HPP
