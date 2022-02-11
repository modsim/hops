#include "KahanSum.hpp"

double hops::kahanSum(vector<double> &summands) {
    double sum = 0.0;
    double c = 0.0;
    for (double summand : summands) {
        double y = summand - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

