// TODO check if headers are complete

#include "Statistics/Autocorrelation.hpp"
#include "Statistics/Covariance.hpp"
#include "Statistics/EffectiveSampleSize.hpp"
#include "Statistics/ExpectedSquaredJumpDistance.hpp"
#include "Statistics/IsConstantChain.hpp"
#include "Statistics/PotentialScaleReductionFactor.hpp"

#include "FileReader/CsvReader.hpp"
#include "FileReader/Hdf5Reader.hpp"
#include "FileReader/SbmlModel.hpp"
#include "FileReader/SbmlReader.hpp"

#include "FileWriter/CsvWriter.hpp"
#include "FileWriter/CsvWriterImpl.hpp"
#include "FileWriter/FileWriter.hpp"
#include "FileWriter/FileWriterFactory.hpp"
#include "FileWriter/FileWriterType.hpp"
#include "FileWriter/Hdf5Writer.hpp"

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

#include "MarkovChain/Proposal/ChordStepDistributions.hpp"
#include "MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp"
#include "MarkovChain/Proposal/CSmMALAProposal.hpp"
#include "MarkovChain/Proposal/DikinEllipsoidCalculator.hpp"
#include "MarkovChain/Proposal/DikinProposal.hpp"
#include "MarkovChain/Proposal/HitAndRunProposal.hpp"
#include "MarkovChain/Proposal/GaussianProposal.hpp"
#include "MarkovChain/Proposal/ProposalFactory.hpp"
#include "MarkovChain/Proposal/Proposal.hpp"
#include "MarkovChain/Proposal/ProposalParameter.hpp"
#include "MarkovChain/Proposal/TruncatedNormalDistribution.hpp"

#include "MarkovChain/Recorder/AcceptanceRateRecorder.hpp"
#include "MarkovChain/Recorder/IsAddMessageAvailabe.hpp"
#include "MarkovChain/Recorder/IsClearRecordsAvailable.hpp"
#include "MarkovChain/Recorder/IsStoreRecordAvailable.hpp"
#include "MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp"
#include "MarkovChain/Recorder/MessageRecorder.hpp"
#include "MarkovChain/Recorder/StateRecorder.hpp"
#include "MarkovChain/Recorder/TimestampRecorder.hpp"

#include "MarkovChain/Tuning/AcceptanceRateTarget.hpp"
#include "MarkovChain/Tuning/AcceptanceRateTuner.hpp"
#include "MarkovChain/Tuning/BinarySearchAcceptanceRateTuner.hpp"
#include "MarkovChain/Tuning/ExpectedSquaredJumpDistanceTarget.hpp"
#include "MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.hpp"
#include "MarkovChain/Tuning/SimpleExpectedSquaredJumpDistanceTuner.hpp"
#include "MarkovChain/Tuning/ThompsonSamplingTuner.hpp"

#include "MarkovChain/IsGetColdnessAvailable.hpp"
#include "MarkovChain/IsGetExchangeAttemptProbabilityAvailable.hpp"
#include "MarkovChain/IsGetStepSizeAvailable.hpp"
#include "MarkovChain/IsResetAcceptanceRateAvailable.hpp"
#include "MarkovChain/IsSetColdnessAvailable.hpp"
#include "MarkovChain/IsSetExchangeAttemptProbabilityAvailable.hpp"
#include "MarkovChain/IsSetFisherWeightAvailable.hpp"
#include "MarkovChain/Proposal/IsSetStepSizeAvailable.hpp"
#include "MarkovChain/MarkovChain.hpp"
#include "MarkovChain/MarkovChainAdapter.hpp"
#include "MarkovChain/MarkovChainAttribute.hpp"
#include "MarkovChain/MarkovChainFactory.hpp"
#include "MarkovChain/MarkovChainType.hpp"
#include "MarkovChain/ModelMixin.hpp"
#include "MarkovChain/StateTransformation.hpp"

#include "Model/DegenerateGaussian.hpp"
#include "Model/Mixture.hpp"
#include "Model/Model.hpp"
#include "Model/Gaussian.hpp"
#include "Model/Rosenbrock.hpp"

#include "Parallel/OpenMPControls.hpp"
#ifdef HOPS_DNEST4_SUPPORT
#include "NestedSampling/DNest4EnvironmentSingleton.hpp"
#include "NestedSampling/DNest4Adapter.hpp"
#endif //HOPS_DNEST4_SUPPORT

#include "Polytope/MaximumVolumeEllipsoid.hpp"
#include "Polytope/NormalizePolytope.hpp"
#include "Polytope/SimplexFactory.hpp"

#include "RandomNumberGenerator/RandomNumberGenerator.hpp"

#include "Transformation/Transformation.hpp"

#include "Utility/MatrixType.hpp"
#include "Utility/VectorType.hpp"

#ifdef HOPS_HEADER_ONLY

#include "FileReader/CsvReader.cpp"
#include "FileReader/SbmlReader.cpp"

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

#include "Utility/Data.cpp"

#ifdef HOPS_HDF5_SUPPORT
#include "FileReader/Hdf5Reader.cpp"
#include "FileWriter/Hdf5Writer.cpp"
#endif //HOPS_HDF5_SUPPORT

#endif //HOPS_HEADER_ONLY
