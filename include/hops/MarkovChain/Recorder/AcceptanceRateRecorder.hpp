#ifndef HOPS_ACCEPTANCERATERECORDER_HPP
#define HOPS_ACCEPTANCERATERECORDER_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <vector>

namespace hops {
    template<typename MarkovChainImpl>
    class AcceptanceRateRecorder : public MarkovChainImpl {
    public:
        explicit AcceptanceRateRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {}

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("acceptanceRates", records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        };

        void storeRecord() {
            records.emplace_back(MarkovChainImpl::getAcceptanceRate());
            if constexpr(IsStoreRecordAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::storeRecord();
            }
        }

        void clearRecords() {
            records.clear();
            if constexpr(IsClearRecordsAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::clearRecords();
            }
        }

    private:
        std::vector<double> records;
    };
}

#endif //HOPS_ACCEPTANCERATERECORDER_HPP
