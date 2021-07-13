#ifndef HOPS_CSVWRITERIMPL_HPP
#define HOPS_CSVWRITERIMPL_HPP

#include <ostream>
#include <vector>

namespace hops::internal {
    class CsvWriterImpl {
    public:
        template<typename Derived>
        static void writeEigenVectorRecords(std::ostream &out, const std::vector<Derived> &records);

        template<typename Derived>
        static void writeOneDimensionalRecords(std::ostream &out, const std::vector<Derived> &records);
    };
}

#endif //HOPS_CSVWRITERIMPL_HPP
