if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            AcceptanceRateRecorder.hpp
            IsAddMessageAvailabe.hpp
            IsClearRecordsAvailable.hpp
            IsStoreRecordAvailable.hpp
            IsWriteRecordsToFileAvailable.hpp
            MessageRecorder.hpp
            StateRecorder.hpp
            TimestampRecorder.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
