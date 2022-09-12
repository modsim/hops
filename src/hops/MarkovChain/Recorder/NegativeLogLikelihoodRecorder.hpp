#ifndef HOPS_NEGATIVELOGLIKELIHOODRECORDER_HPP
#define HOPS_NEGATIVELOGLIKELIHOODRECORDER_HPP

#include "../../FileWriter/FileWriter.hpp"
#include "IsClearRecordsAvailable.hpp"
#include "IsStoreRecordAvailable.hpp"
#include "IsWriteRecordsToFileAvailable.hpp"

#include <memory>
#include <vector>

namespace hops {
    template<typename MarkovChainImpl>
    class NegativeLogLikelihoodRecorder : public MarkovChainImpl {
    public:
        explicit NegativeLogLikelihoodRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {
            records = std::make_shared<std::vector<double>>();
        }

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("negativeLogLikelihood", *records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        }

        void storeRecord() {
            records->emplace_back(MarkovChainImpl::getStateNegativeLogLikelihood());
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

#endif //HOPS_NEGATIVELOGLIKELIHOODRECORDER_HPP
