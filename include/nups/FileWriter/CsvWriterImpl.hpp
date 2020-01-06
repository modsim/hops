#ifndef NUPS_CSVWRITERIMPL_HPP
#define NUPS_CSVWRITERIMPL_HPP

#include <ostream>
#include <vector>

namespace nups::internal {
    class CsvWriterImpl {
    public:
        template<typename Derived>
        static void writeEigenVectorRecords(std::ostream &out, const std::vector<Derived> &records);

        template<typename Derived>
        static void writeOneDimensionalRecords(std::ostream &out, const std::vector<Derived> &records);
    };
}


#endif //NUPS_CSVWRITERIMPL_HPP
