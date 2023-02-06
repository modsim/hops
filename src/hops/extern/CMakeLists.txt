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

if(HOPS_DNEST4_SUPPORT)
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
		DNest4/code/Pybind11_abortable.hpp
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
endif(HOPS_DNEST4_SUPPORT)
