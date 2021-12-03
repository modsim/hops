#include <algorithm>
#include <cctype>

#include "StringUtility.hpp"

std::string hops::toLowerCase(const std::string &str) {
    std::string lowerCaseString = str;
    std::transform(str.begin(), str.end(), lowerCaseString.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lowerCaseString;
}

