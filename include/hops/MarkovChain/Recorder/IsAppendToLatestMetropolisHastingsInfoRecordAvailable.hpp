#ifndef HOPS_ISSTOREMETROPOLISHASTINGSINFOAVAILABLE_HPP
#define HOPS_ISSTOREMETROPOLISHASTINGSINFOAVAILABLE_HPP

#include <string>
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsAppendToLatestMetropolisHastingsInfoRecordAvailable : std::false_type {
    };

    template<typename T>
    struct IsAppendToLatestMetropolisHastingsInfoRecordAvailable<T, std::void_t<decltype(std::declval<T>().appendToLatestMetropolisHastingsInfoRecord(
            std::declval<const std::string &>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISSTOREMETROPOLISHASTINGSINFOAVAILABLE_HPP
