#ifndef HOPS_ISSTOREMETROPOLISHASTINGSINFORECORDAVAILABLE_HPP
#define HOPS_ISSTOREMETROPOLISHASTINGSINFORECORDAVAILABLE_HPP

#include <string>
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsStoreMetropolisHastingsInfoRecordAvailable : std::false_type {
    };

    template<typename T>
    struct IsStoreMetropolisHastingsInfoRecordAvailable<T, std::void_t<decltype(std::declval<T>().storeMetropolisHastingsInfoRecord(
            std::declval<const std::string&>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISSTOREMETROPOLISHASTINGSINFORECORDAVAILABLE_HPP
