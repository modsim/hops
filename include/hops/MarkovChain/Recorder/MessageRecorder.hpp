#ifndef HOPS_MESSAGERECORDER_HPP
#define HOPS_MESSAGERECORDER_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <string>
#include <vector>

namespace hops {
    template<typename MarkovChainImpl>
    class MessageRecorder : public MarkovChainImpl {
    public:
        explicit MessageRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {}

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("Messages", records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        };

        void storeRecord() {
            records.emplace_back(latestMessage);
            latestMessage = "";
            if constexpr(IsStoreRecordAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::storeRecord();
            }
        }

        void addMessage(const std::string &message) {
            latestMessage += message;
        }

        void clearRecords() {
            records.clear();
        }

    private:
        std::vector<std::string> records;
        std::string latestMessage;
    };
}

#endif //HOPS_MESSAGERECORDER_HPP