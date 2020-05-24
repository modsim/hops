if(WIN32)
    # MS-MPI refuses to work, so we skip it for windows for now.
#  #This is for finding MS-MPI.
#  #set(_MPI_PREFIX_PATH)
#  #list(APPEND _MPI_PREFIX_PATH "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\MPI;Path]/..")
#  set(MPIEXEC $ENV{MSMPI_BIN}/mpiexec.exe)
#  # For building MPI programs the selected Visual Studio compiler is used, namely cl.exe.
#  # So there is no need to set a specific MPI compiler.
#  #set(MPI_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
#  set(MPI_CXX_INCLUDE_PATH $ENV{MSMPI_INC})
#  # Make sure the correct libraries (64-bit or 32-bit) are selected.
#  # Decide between 32-bit and 64-bit libraries for Microsoft's MPI
#  if("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
#    set(MS_MPI_ARCH_DIR x64)
#  else()
#    set(MS_MPI_ARCH_DIR x86)
#  endif()
#  # set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/${MS_MPI_ARCH_DIR}/msmpi.lib")
#  set(MPI_CXX_LIBRARIES "$ENV{MSMPI_LIB64}/msmpi.lib")
#  set(MPI_C_INCLUDE_PATH ${MPI_CXX_INCLUDE_PATH})
#  set(MPI_C_LIBRARIES ${MPI_CXX_LIBRARIES})
#  #set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "Flag used by MPI to specify
##the number of processes for MPIEXEC; the next option will be the number of
##processes.")
#
##set(CMAKE_C_FLAGS "${CMAKE_FLAGS} ${MPI_FLAGS}")
##set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
##set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS}")
#    target_include_directories(hops PUBLIC ${MPI_C_INCLUDE_PATH})
#    target_link_libraries(hops PUBLIC ${MPI_C_LIBRARIES})
    message(WARNING "MPI is currently not supported on current platform.")
else()
	find_package(MPI REQUIRED)
    target_compile_definitions(hops PUBLIC HOPS_MPI_SUPPORTED)
    target_include_directories(hops PUBLIC ${MPI_INCLUDE_PATH})
    target_link_libraries(hops PUBLIC ${MPI_LIBRARIES})
    target_sources(hops PRIVATE ColdnessAttribute.hpp ParallelTempering.hpp)
endif()

