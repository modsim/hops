#include "FileReader/CsvReader.hpp"

#include "FileWriter/CsvWriter.hpp"
#include "FileWriter/CsvWriterImpl.hpp"
#include "FileWriter/FileWriter.hpp"
#include "FileWriter/FileWriterFactory.hpp"
#include "FileWriter/FileWriterType.hpp"

#include "LinearProgram/LinearProgram.hpp"
#include "LinearProgram/LinearProgramFactory.hpp"
#include "LinearProgram/LinearProgramSolution.hpp"
#include "LinearProgram/LinearProgramStatus.hpp"

#include "MarkovChain/Draw/IsAcceptProposalAvailable.hpp"
#include "MarkovChain/Draw/IsCalculateLogAcceptanceProbabilityAvailable.hpp"
#include "MarkovChain/Draw/MetropolisHastingsFilter.hpp"
#include "MarkovChain/Draw/NoOpDrawAdapter.hpp"

#include "MarkovChain/ParallelTempering/Coldness.hpp"
#include "MarkovChain/ParallelTempering/ParallelTempering.hpp"

#include "MarkovChain/Proposal/AdaptiveMetropolisProposal.hpp"
#include "MarkovChain/Proposal/BallWalkProposal.hpp"
#include "MarkovChain/Proposal/BilliardAdaptiveMetropolisProposal.hpp"
#include "MarkovChain/Proposal/BilliardMALAProposal.hpp"
#include "MarkovChain/Proposal/BilliardWalkProposal.hpp"
#include "MarkovChain/Proposal/ChordStepDistributions.hpp"
#include "MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp"
#include "MarkovChain/Proposal/CSmMALAProposal.hpp"
#include "MarkovChain/Proposal/DikinEllipsoidCalculator.hpp"
#include "MarkovChain/Proposal/DikinProposal.hpp"
#include "MarkovChain/Proposal/GaussianProposal.hpp"
#include "MarkovChain/Proposal/HitAndRunProposal.hpp"
#include "MarkovChain/Proposal/IsSetStepSizeAvailable.hpp"
#include "MarkovChain/Proposal/ProposalFactory.hpp"
#include "MarkovChain/Proposal/Proposal.hpp"
#include "MarkovChain/Proposal/ProposalParameter.hpp"
#include "MarkovChain/Proposal/Reflector.hpp"
#include "MarkovChain/Proposal/ReversibleJumpProposal.hpp"
#include "MarkovChain/Proposal/TruncatedGaussianProposal.hpp"
#include "MarkovChain/Proposal/TruncatedNormalDistribution.hpp"

#include "MarkovChain/Recorder/AcceptanceRateRecorder.hpp"
#include "MarkovChain/Recorder/IsAddMessageAvailabe.hpp"
#include "MarkovChain/Recorder/IsClearRecordsAvailable.hpp"
#include "MarkovChain/Recorder/IsStoreRecordAvailable.hpp"
#include "MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp"
#include "MarkovChain/Recorder/MessageRecorder.hpp"
#include "MarkovChain/Recorder/NegativeLogLikelihoodRecorder.hpp"
#include "MarkovChain/Recorder/StateRecorder.hpp"
#include "MarkovChain/Recorder/TimestampRecorder.hpp"

#include "MarkovChain/Tuning/AcceptanceRateTarget.hpp"
#include "MarkovChain/Tuning/AcceptanceRateTuner.hpp"
#include "MarkovChain/Tuning/BinarySearchAcceptanceRateTuner.hpp"
#include "MarkovChain/Tuning/ExpectedSquaredJumpDistanceTarget.hpp"
#include "MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.hpp"
#include "MarkovChain/Tuning/GridSearchTuner.hpp"
#include "MarkovChain/Tuning/SimpleExpectedSquaredJumpDistanceTuner.hpp"
#include "MarkovChain/Tuning/ThompsonSamplingTuner.hpp"
#include "MarkovChain/Tuning/TuningTarget.hpp"

#include "MarkovChain/MarkovChain.hpp"
#include "MarkovChain/MarkovChainAdapter.hpp"
#include "MarkovChain/MarkovChainFactory.hpp"
#include "MarkovChain/MarkovChainType.hpp"
#include "MarkovChain/ModelMixin.hpp"
#include "MarkovChain/ModelWrapper.hpp"
#include "MarkovChain/StateTransformation.hpp"

#include "Model/DegenerateGaussian.hpp"
#include "Model/Gaussian.hpp"
#include "Model/Mixture.hpp"
#include "Model/Model.hpp"
#include "Model/JumpableModel.hpp"
#include "Model/Rosenbrock.hpp"

#ifdef HOPS_DNEST4_SUPPORT
#include "NestedSampling/DNest4EnvironmentSingleton.hpp"
#include "NestedSampling/DNest4Adapter.hpp"

#include "extern/DNest4.hpp"
#include "extern/DNest4/code/Barrier.h"
#include "extern/DNest4/code/CommandLineOptions.h"
#include "extern/DNest4/code/Distributions/Cauchy.h"
#include "extern/DNest4/code/Distributions/ContinuousDistribution.h"
#include "extern/DNest4/code/Distributions/Exponential.h"
#include "extern/DNest4/code/Distributions/Gaussian.h"
#include "extern/DNest4/code/Distributions/Jeffreys.h"
#include "extern/DNest4/code/Distributions/Kumaraswamy.h"
#include "extern/DNest4/code/Distributions/Laplace.h"
#include "extern/DNest4/code/Distributions/LogUniform.h"
#include "extern/DNest4/code/Distributions/Pareto.h"
#include "extern/DNest4/code/Distributions/Rayleigh.h"
#include "extern/DNest4/code/Distributions/Triangular.h"
#include "extern/DNest4/code/Distributions/Uniform.h"
#include "extern/DNest4/code/DNest4.h"
#include "extern/DNest4/code/Level.h"
#include "extern/DNest4/code/LikelihoodType.h"
#include "extern/DNest4/code/Options.h"
#include "extern/DNest4/code/Pybind11_abortable.hpp"
#include "extern/DNest4/code/RNG.h"
#include "extern/DNest4/code/Sampler.h"
#include "extern/DNest4/code/SamplerImpl.h"
#include "extern/DNest4/code/Start.h"
#include "extern/DNest4/code/StartImpl.h"
#include "extern/DNest4/code/Utils.h"
#include "extern/DNest4/code/Version.h"

#endif //HOPS_DNEST4_SUPPORT

#include "Optimization/GaussianProcess.hpp"
#include "Optimization/ThompsonSampling.hpp"

#include "Polytope/MaximumVolumeEllipsoid.hpp"
#include "Polytope/NormalizePolytope.hpp"
#include "Polytope/SimplexFactory.hpp"

#include "RandomNumberGenerator/RandomNumberGenerator.hpp"

#include "Statistics/Autocorrelation.hpp"
#include "Statistics/Covariance.hpp"
#include "Statistics/ExpectedSquaredJumpDistance.hpp"
#include "Statistics/IsConstantChain.hpp"


#include "Transformation/LinearTransformation.hpp"
#include "Transformation/Transformation.hpp"

#include "Utility/DefaultDimensionNames.hpp"
#include "Utility/HopsWithinHopsy.hpp"
#include "Utility/KahanSum.hpp"
#include "Utility/LogSqrtDeterminant.hpp"
#include "Utility/MatrixType.hpp"
#include "Utility/Sampling.hpp"
#include "Utility/StringUtility.hpp"
#include "Utility/VectorType.hpp"

#ifdef HOPS_HEADER_ONLY


#include "FileReader/CsvReader.cpp"
#include "FileReader/SbmlReader.cpp"

#ifdef HOPS_HDF5_SUPPORT
#include "FileReader/Hdf5Reader.cpp"
#include "FileWriter/Hdf5Writer.cpp"
#endif //HOPS_HDF5_SUPPORT

#include "FileWriter/CsvWriter.cpp"
#include "FileWriter/CsvWriterImpl.cpp"
#include "FileWriter/FileWriterFactory.cpp"

#include "LinearProgram/GurobiEnvironmentSingleton.cpp"
#include "LinearProgram/LinearProgram.cpp"
#include "LinearProgram/LinearProgramClpImpl.cpp"
#include "LinearProgram/LinearProgramGurobiImpl.cpp"

#include "MarkovChain/Tuning/BinarySearchAcceptanceRateTuner.cpp"
#include "MarkovChain/Tuning/AcceptanceRateTuner.cpp"
#include "MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.cpp"
#include "MarkovChain/Tuning/SimpleExpectedSquaredJumpDistanceTuner.cpp"

#include "Polytope/MaximumVolumeEllipsoid.cpp"

#include "Utility/DefaultDimensionNames.cpp"
#include "Utility/KahanSum.cpp"
#include "Utility/Sampling.cpp"
#include "Utility/StringUtility.hpp"


#endif //HOPS_HEADER_ONLY
