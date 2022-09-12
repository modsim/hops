#ifndef HOPS_TIMESTAMPRECORDER_HPP
#define HOPS_TIMESTAMPRECORDER_HPP

#include "../../FileWriter/FileWriter.hpp"
#include "IsClearRecordsAvailable.hpp"
#include "IsStoreRecordAvailable.hpp"
#include "IsWriteRecordsToFileAvailable.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hops {
    template<typename MarkovChainImpl>
    class TimestampRecorder : public MarkovChainImpl {
    public:
        explicit TimestampRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {
            records = std::make_shared<std::vector<long>>();
        }

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("timestamps", *records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        };

        [[nodiscard]] std::vector<long> getTimestampRecords() const {
            return *records;
        }

        void storeRecord() {
            records->emplace_back(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()
                    ).count()
            );

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
        std::shared_ptr<std::vector<long>> records;
    };
}

#endif //HOPS_TIMESTAMPRECORDER_HPP
