#ifndef HOPS_NEGATIVELOGLIKELIHOODRECORDER_HPP
#define HOPS_NEGATIVELOGLIKELIHOODRECORDER_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <vector>

namespace hops {
    template<typename MarkovChainImpl>
    class NegativeLogLikelihoodRecorder : public MarkovChainImpl {
    public:
        explicit NegativeLogLikelihoodRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(
                markovChainImpl) {}

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("negativeLogLikelihood", records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        }

        void storeRecord() {
            records.emplace_back(MarkovChainImpl::getNegativeLogLikelihoodOfCurrentState());
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

#endif //HOPS_NEGATIVELOGLIKELIHOODRECORDER_HPP
