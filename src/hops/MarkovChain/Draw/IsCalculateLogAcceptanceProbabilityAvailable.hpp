#ifndef HOPS_ISCALCULATELOGACCEPTANCEPROBABILITYAVAILABLE_HPP
#define HOPS_ISCALCULATELOGACCEPTANCEPROBABILITYAVAILABLE_HPP

#include <type_traits>

namespace hops {
    /**
     * @Brief deprecated. TODO remove
     * @tparam T
     */
    template<typename T, typename = void>
    struct IsCalculateLogAcceptanceProbabilityAvailable : std::false_type {
    };

    template<typename T>
    struct IsCalculateLogAcceptanceProbabilityAvailable<T, std::void_t<decltype(std::declval<T>().computeLogAcceptanceProbability())> > :
            std::true_type {
    };
}

#endif //HOPS_ISCALCULATELOGACCEPTANCEPROBABILITYAVAILABLE_HPP
