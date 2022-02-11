#ifndef HOPS_WITHIN_HOPSY_HPP
#define HOPS_WITHIN_HOPSY_HPP

#ifdef HOPS_WITHIN_HOPSY

#include <pybind11/detail/common.h>

#define ABORTABLE if (PyErr_CheckSignals() != 0) throw pybind11::error_already_set();

#else

#define ABORTABLE 

#endif

#endif // HOPS_WITHIN_HOPSY_HPP
