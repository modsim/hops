#ifndef HOPS_OPENMPCONTROLS_HPP
#define HOPS_OPENMPCONTROLS_HPP

#ifdef _OPENMP
#include <omp.h>
#endif 

namespace hops {
#ifdef _OPENMP
    static int numberOfThreads = omp_get_max_threads();
#else
    static int numberOfThreads = 1;
#endif 
}

#endif // HOPS_OPENMPCONTROLS_HPP
