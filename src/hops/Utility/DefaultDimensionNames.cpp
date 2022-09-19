#include "DefaultDimensionNames.hpp"

std::vector<std::string> hops::createDefaultDimensionNames(long numberOfDimensions, const std::string &prefix) {
    // Default implementation sets names as x_i for dimension i
    std::vector<std::string> names;
    for (long i = 0; i < numberOfDimensions; ++i) {
        names.emplace_back(prefix + std::to_string(i));
    }
    return names;
}
