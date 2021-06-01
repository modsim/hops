#include "Diagnostics/IsConstantChain.hpp"
#include "Diagnostics/Autocorrelation.hpp"
#include "Diagnostics/EffectiveSampleSize.hpp"
#include "Diagnostics/PotentialScaleReductionFactor.hpp"
#include "Diagnostics/ExpectedSquaredJumpDistance.hpp"

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

#include "MarkovChain/ParallelTempering/ColdnessAttribute.hpp"
#include "MarkovChain/ParallelTempering/ParallelTempering.hpp"

#include "MarkovChain/Proposal/ChordStepDistributions.hpp"
#include "MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp"
#include "MarkovChain/Proposal/DikinEllipsoidCalculator.hpp"
#include "MarkovChain/Proposal/DikinProposal.hpp"
#include "MarkovChain/Proposal/HitAndRunProposal.hpp"
#include "MarkovChain/Proposal/TruncatedNormalDistribution.hpp"

#include "MarkovChain/Recorder/AcceptanceRateRecorder.hpp"
#include "MarkovChain/Recorder/IsAddMessageAvailabe.hpp"
#include "MarkovChain/Recorder/IsClearRecordsAvailable.hpp"
#include "MarkovChain/Recorder/IsStoreRecordAvailable.hpp"
#include "MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp"
#include "MarkovChain/Recorder/MessageRecorder.hpp"
#include "MarkovChain/Recorder/StateRecorder.hpp"
#include "MarkovChain/Recorder/TimestampRecorder.hpp"

#include "MarkovChain/AcceptanceRateTuner.hpp"
#include "MarkovChain/ExpectedSquaredJumpDistanceTuner.hpp"
#include "MarkovChain/IsGetColdnessAvailable.hpp"
#include "MarkovChain/IsGetExchangeAttemptProbabilityAvailable.hpp"
#include "MarkovChain/IsGetStepSizeAvailable.hpp"
#include "MarkovChain/IsResetAcceptanceRateAvailable.hpp"
#include "MarkovChain/IsSetColdnessAvailable.hpp"
#include "MarkovChain/IsSetExchangeAttemptProbabilityAvailable.hpp"
#include "MarkovChain/IsSetFisherWeightAvailable.hpp"
#include "MarkovChain/IsSetStepSizeAvailable.hpp"
#include "MarkovChain/MarkovChain.hpp"
#include "MarkovChain/MarkovChainAdapter.hpp"
#include "MarkovChain/MarkovChainAttribute.hpp"
#include "MarkovChain/MarkovChainFactory.hpp"
#include "MarkovChain/MarkovChainType.hpp"
#include "MarkovChain/StateTransformation.hpp"

#include "Model/DegenerateMultivariateGaussianModel.hpp"
#include "Model/DynMultimodalModel.hpp"
#include "Model/ModelMixin.hpp"
#include "Model/MultimodalModel.hpp"
#include "Model/MultivariateGaussianModel.hpp"
#include "Model/RosenbrockModel.hpp"
#include "Model/UniformDummyModel.hpp"

#include "Polytope/MaximumVolumeEllipsoid.hpp"
#include "Polytope/NormalizePolytope.hpp"
#include "Polytope/SimplexFactory.hpp"

#include "RandomNumberGenerator/RandomNumberGenerator.hpp"

#include "Transformation/Transformation.hpp"

#include "Utility/ChainData.hpp"
#include "Utility/Data.hpp"
#include "Utility/Exceptions.hpp"
#include "Utility/Problem.hpp"
#include "Utility/Run.hpp"

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

#include "MarkovChain/AcceptanceRateTuner.cpp"
#include "MarkovChain/ExpectedSquaredJumpDistanceTuner.cpp"

#include "Polytope//MaximumVolumeEllipsoid.cpp"

#ifdef HOPS_HDF5_SUPPORT
#include "FileReader/Hdf5Reader.cpp"
#include "FileWriter/Hdf5Writer.cpp"
#endif //HOPS_HDF5_SUPPORT

#endif //HOPS_HEADER_ONLY
