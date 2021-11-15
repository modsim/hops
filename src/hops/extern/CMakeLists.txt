########################################################################################################################
# pcg64
########################################################################################################################

target_sources(hops PRIVATE
        pcg-cpp/pcg_extras.hpp
        pcg-cpp/pcg_random.hpp
        pcg-cpp/pcg_uint128.hpp
        )

########################################################################################################################
# DNest44
########################################################################################################################

target_sources(hops PRIVATE
        DNest4.hpp
        DNest4/code/Barrier.cpp
        DNest4/code/Barrier.h
        DNest4/code/CommandLineOptions.cpp
        DNest4/code/CommandLineOptions.h
        DNest4/code/Distributions/Cauchy.cpp
        DNest4/code/Distributions/Cauchy.h
        DNest4/code/Distributions/ContinuousDistribution.cpp
        DNest4/code/Distributions/ContinuousDistribution.h
        DNest4/code/Distributions/Exponential.cpp
        DNest4/code/Distributions/Exponential.h
        DNest4/code/Distributions/Gaussian.cpp
        DNest4/code/Distributions/Gaussian.h
        DNest4/code/Distributions/Jeffreys.cpp
        DNest4/code/Distributions/Jeffreys.h
        DNest4/code/Distributions/Kumaraswamy.cpp
        DNest4/code/Distributions/Kumaraswamy.h
        DNest4/code/Distributions/Laplace.cpp
        DNest4/code/Distributions/Laplace.h
        DNest4/code/Distributions/LogUniform.cpp
        DNest4/code/Distributions/LogUniform.h
        DNest4/code/Distributions/Pareto.cpp
        DNest4/code/Distributions/Pareto.h
        DNest4/code/Distributions/Rayleigh.cpp
        DNest4/code/Distributions/Rayleigh.h
        DNest4/code/Distributions/Triangular.cpp
        DNest4/code/Distributions/Triangular.h
        DNest4/code/Distributions/Uniform.cpp
        DNest4/code/Distributions/Uniform.h
        DNest4/code/DNest4.h
        DNest4/code/Level.cpp
        DNest4/code/Level.h
        DNest4/code/LikelihoodType.cpp
        DNest4/code/LikelihoodType.h
        DNest4/code/Options.cpp
        DNest4/code/Options.h
        DNest4/code/RNG.cpp
        DNest4/code/RNG.h
        DNest4/code/Sampler.h
        DNest4/code/SamplerImpl.h
        DNest4/code/Start.h
        DNest4/code/StartImpl.h
        DNest4/code/Utils.cpp
        DNest4/code/Utils.h
        DNest4/code/Version.h
        )

########################################################################################################################
# HighFive
########################################################################################################################

find_package(HDF5 COMPONENTS C CXX)

if (HDF5_FOUND AND HOPS_HDF5_SUPPORT)
    message(STATUS "Found HDF5 libraries.")
    target_include_directories(hops INTERFACE ${HDF5_INCLUDE_DIRS})
    target_include_directories(hops ${SCOPE} ${HDF5_INCLUDE_DIRS})
    target_link_libraries(hops INTERFACE ${HDF5_LIBRARIES})
    target_link_libraries(hops ${SCOPE} ${HDF5_LIBRARIES})
    target_compile_definitions(hops INTERFACE HOPS_HDF5_FOUND)
    target_compile_definitions(hops ${SCOPE} HOPS_HDF5_FOUND)
    set(HIGHFIVE_USE_EIGEN ON)
    set(HIGHFIVE_USE_BOOST ON)
    set(HIGHFIVE_UNIT_TESTS OFF)
    set(HIGHFIVE_EXAMPLES OFF)

    add_subdirectory(HighFive)
    add_dependencies(hops HighFive)
    target_link_libraries(hops INTERFACE HighFive)
    target_link_libraries(hops ${SCOPE} HighFive)

else (HDF5_FOUND AND HOPS_HDF5_SUPPORT)
    message(STATUS "HDF5 disabled or libraries not found. Continuing without HDF5 support.")
endif (HDF5_FOUND AND HOPS_HDF5_SUPPORT)
