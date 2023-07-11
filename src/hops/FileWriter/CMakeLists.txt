if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops
            PRIVATE
            CsvWriter.hpp
            CsvWriter.cpp
            CsvWriterImpl.cpp
            CsvWriterImpl.hpp
            FileWriter.hpp
            FileWriterFactory.hpp
            FileWriterFactory.cpp
            FileWriterType.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
