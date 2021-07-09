#ifndef HOPS_ISCALCULATELOGLIKELIHOODGRADIENTAVAILABLE_HPP
#define HOPS_ISCALCULATELOGLIKELIHOODGRADIENTAVAILABLE_HPP

#include <string>
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsCalculateLogLikelihoodGradientAvailable : std::false_type {
    };

    template<typename T>
    struct IsCalculateLogLikelihoodGradientAvailable<T, std::void_t<decltype(std::declval<T>().computeLogLikelihoodGradient(
            std::declval<const typename T::VectorType &>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISCALCULATELOGLIKELIHOODGRADIENTAVAILABLE_HPP
