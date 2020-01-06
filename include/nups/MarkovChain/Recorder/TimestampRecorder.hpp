#ifndef NUPS_TIMESTAMPRECORDER_HPP
#define NUPS_TIMESTAMPRECORDER_HPP

#include <chrono>
#include <nups/FileWriter/FileWriter.hpp>
#include <nups/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <nups/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <nups/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <vector>

namespace nups {
    template<typename MarkovChainImpl>
    class TimestampRecorder : public MarkovChainImpl {
    public:
        explicit TimestampRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {}

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("timestamps", records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        };

        void storeRecord() {
            records.emplace_back(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()
                    ).count()
            );

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
        std::vector<long> records;
    };
}

#endif //NUPS_TIMESTAMPRECORDER_HPP
