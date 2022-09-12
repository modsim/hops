if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            HopsWithinHopsy.hpp
            KahanSum.hpp
            LogSqrtDeterminant.hpp
            KahanSum.cpp
            MatrixType.hpp
            Sampling.hpp
            Sampling.cpp
            StringUtility.hpp
            StringUtility.cpp
            VectorType.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
