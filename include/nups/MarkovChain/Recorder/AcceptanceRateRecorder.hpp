#ifndef NUPS_ACCEPTANCERATERECORDER_HPP
#define NUPS_ACCEPTANCERATERECORDER_HPP

#include <nups/FileWriter/FileWriter.hpp>
#include <nups/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <nups/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <nups/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <vector>

namespace nups {
    template<typename MarkovChainImpl>
    class AcceptanceRateRecorder {
    public:
        explicit AcceptanceRateRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {}

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("acceptance rates", records);
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

#endif //NUPS_ACCEPTANCERATERECORDER_HPP
