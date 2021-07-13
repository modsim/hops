#ifndef HOPS_ISINSTALLDATAOBJECTAVAILABLE_HPP
#define HOPS_ISINSTALLDATAOBJECTAVAILABLE_HPP

#include "../../Utility/ChainData.hpp"
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsInstallDataObjectAvailable : std::false_type {
    };

    template<typename T>
    struct IsInstallDataObjectAvailable<T, 
            std::void_t<decltype(std::declval<T>().installDataObject(
                        std::declval<ChainData&>()
            ))> > :
            std::true_type {
    };
}

#endif //HOPS_ISINSTALLDATAOBJECTAVAILABLE_HPP
