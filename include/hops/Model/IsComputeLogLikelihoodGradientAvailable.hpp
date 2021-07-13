#ifndef HOPS_ISCOMPUTELOGLIKELIHOODGRADIENTAVAILABLE_HPP
#define HOPS_ISCOMPUTELOGLIKELIHOODGRADIENTAVAILABLE_HPP

#include <string>
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsComputeLogLikelihoodGradientAvailable : std::false_type {
    };

    template<typename T>
    struct IsComputeLogLikelihoodGradientAvailable<T, std::void_t<decltype(std::declval<T>().computeLogLikelihoodGradient(
            std::declval<const typename T::VectorType &>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISCOMPUTELOGLIKELIHOODGRADIENTAVAILABLE_HPP
