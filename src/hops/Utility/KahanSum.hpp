#ifndef HOPS_KAHANSUM_HPP
#define HOPS_KAHANSUM_HPP

#include <vector>

namespace hops {
    using namespace std;

    /**
     * @brief numerically stabler sum when number of summands is large
     * @details Reference https://www.geeksforgeeks.org/kahan-summation-algorithm/
     * @param summands
     * @return
     */
    double kahanSum(vector<double> &summands);
}

#endif //HOPS_KAHANSUM_HPP
