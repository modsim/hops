if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            DefaultDimensionNames.hpp
            DefaultDimensionNames.cpp
            HopsWithinHopsy.hpp
            KahanSum.hpp
            LogSqrtDeterminant.hpp
            KahanSum.cpp
            MatrixType.hpp
            StringUtility.hpp
            StringUtility.cpp
            VectorType.hpp
    )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")

if (HOPS_TUNING)
    if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
        target_sources(hops PRIVATE
                Sampling.hpp
                Sampling.cpp
        )
    endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
endif (HOPS_TUNING)
