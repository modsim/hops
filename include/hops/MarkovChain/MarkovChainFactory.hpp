#ifndef HOPS_MARKOVCHAINFACTORY_HPP
#define HOPS_MARKOVCHAINFACTORY_HPP

#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainType.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Draw/NoOpDrawAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/ParallelTempering/ColdnessAttribute.hpp>
#include <hops/MarkovChain/ParallelTempering/ParallelTempering.hpp>
#include <hops/MarkovChain/Proposal/BallWalkProposal.hpp>
#include <hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <hops/MarkovChain/Proposal/CSmMALANoGradientProposal.hpp>
#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>
#include <hops/MarkovChain/Recorder/AcceptanceRateRecorder.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Recorder/TimestampRecorder.hpp>
#include <hops/MarkovChain/StateTransformation.hpp>
#include <hops/Model/ModelMixin.hpp>
#include <hops/Transformation/Transformation.hpp>
#include <hops/MarkovChain/Recorder/NegativeLogLikelihoodRecorder.hpp>

namespace hops {
    class MarkovChainFactory {
    public:
        /**
         * @brief Creates a Markov chain for uniform sampling of convex polytopes.
         * @tparam MatrixType
         * @tparam VectorType
         * @param type
         * @param inequalityLhs
         * @param inequalityRhs
         * @param startingPoint
         * @return
         */
        template<typename MatrixType, typename VectorType>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType inequalityLhs,
                VectorType inequalityRhs,
                VectorType startingPoint
        ) {
            switch (type) {
                case MarkovChainType::BallWalk : {
                    return addRecordersAndAdapter(
                            NoOpDrawAdapter(
                                    BallWalkProposal(
                                            Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                    inequalityLhs), inequalityRhs, startingPoint)
                            )
                    );
                }
                case MarkovChainType::CoordinateHitAndRun : {
                    return addRecordersAndAdapter(
                            NoOpDrawAdapter(
                                    CoordinateHitAndRunProposal(
                                            Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                    inequalityLhs), inequalityRhs, startingPoint)
                            )
                    );
                }
                case MarkovChainType::DikinWalk : {
                    return addRecordersAndAdapter(
                            MetropolisHastingsFilter(
                                    DikinProposal(inequalityLhs, inequalityRhs, startingPoint)
                            )
                    );
                }
                case MarkovChainType::HitAndRun: {
                    return addRecordersAndAdapter(
                            NoOpDrawAdapter(
                                    HitAndRunProposal(
                                            Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                    inequalityLhs), inequalityRhs, startingPoint)
                            )
                    );
                }
                default: {
                    throw std::runtime_error("MarkovChainType not supported for uniform sampling.");
                }
            }
        }

        /**
         * @brief Creates a Markov chain for uniform sampling of rounded convex polytopes.
         * @tparam MatrixType
         * @tparam VectorType
         * @param type
         * @param roundedInequalityLhs
         * @param roundedInequalityRhs
         * @param startingPoint
         * @param unroundingTransformation
         * @param unroundingShift
         * @return
         */
        template<typename MatrixType, typename VectorType>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType roundedInequalityLhs,
                VectorType roundedInequalityRhs,
                VectorType startingPoint,
                MatrixType unroundingTransformation,
                VectorType unroundingShift
        ) {
            switch (type) {
                case MarkovChainType::BallWalk : {
                    return addRecordersAndAdapter(
                            NoOpDrawAdapter(
                                    StateTransformation(
                                            BallWalkProposal(
                                                    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                            roundedInequalityLhs),
                                                    roundedInequalityRhs,
                                                    startingPoint),
                                            Transformation(unroundingTransformation, unroundingShift)
                                    )
                            )
                    );
                }
                case MarkovChainType::CoordinateHitAndRun : {
                    return addRecordersAndAdapter(
                            NoOpDrawAdapter(
                                    StateTransformation(
                                            CoordinateHitAndRunProposal(
                                                    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                            roundedInequalityLhs),
                                                    roundedInequalityRhs,
                                                    startingPoint),
                                            Transformation(unroundingTransformation, unroundingShift)
                                    )
                            )
                    );
                }
                case MarkovChainType::DikinWalk : {
                    return addRecordersAndAdapter(
                            MetropolisHastingsFilter(
                                    StateTransformation(
                                            DikinProposal(
                                                    roundedInequalityLhs,
                                                    roundedInequalityRhs,
                                                    startingPoint),
                                            Transformation(unroundingTransformation, unroundingShift)
                                    )
                            )
                    );
                }
                case MarkovChainType::HitAndRun: {
                    return addRecordersAndAdapter(
                            NoOpDrawAdapter(
                                    StateTransformation(
                                            HitAndRunProposal(
                                                    Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                            roundedInequalityLhs),
                                                    roundedInequalityRhs,
                                                    startingPoint),
                                            Transformation(unroundingTransformation, unroundingShift)
                                    )
                            )
                    );
                }
                default: {
                    throw std::runtime_error("MarkovChainType not supported for uniform sampling.");
                }
            }
        }

        /**
         * @brief Creats a Markov chain for sampling the likelihood of a model with the domain of a convex polytope.
         * @tparam MatrixType
         * @tparam VectorType
         * @tparam Model
         * @param type
         * @param inequalityLhs
         * @param inequalityRhs
         * @param startingPoint
         * @param model
         * @return
         */
        template<typename MatrixType, typename VectorType, typename Model>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType inequalityLhs,
                VectorType inequalityRhs,
                VectorType startingPoint,
                Model model,
                bool useParallelTempering
        ) {
            switch (type) {
                case MarkovChainType::BallWalk : {
                    return addRecordersAndAdapter(
                        MetropolisHastingsFilter(
                            ModelMixin(
                                BallWalkProposal<
                                        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                        decltype(inequalityRhs)
                                    >(inequalityLhs, inequalityRhs, startingPoint),
                                ColdnessAttribute(model)
                            )
                        ),
                        useParallelTempering
                    );
                }
                case MarkovChainType::CoordinateHitAndRun : {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            ModelMixin(
                                                    CoordinateHitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                            decltype(inequalityRhs),
                                                            GaussianStepDistribution<typename decltype(inequalityRhs)::Scalar>>(
                                                            inequalityLhs, inequalityRhs, startingPoint),
                                                    ColdnessAttribute(model)
                                            )
                                    )
                            ),
                            useParallelTempering
                    );
                }
                case MarkovChainType::CSmMALA: {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            CSmMALAProposal(ColdnessAttribute(model),
                                                            inequalityLhs,
                                                            inequalityRhs,
                                                            startingPoint)
                                    )
                            ),
                            useParallelTempering
                    );
                }
                case MarkovChainType::CSmMALANoGradient: {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            CSmMALANoGradientProposal(ColdnessAttribute(model),
                                                                      inequalityLhs,
                                                                      inequalityRhs,
                                                                      startingPoint)
                                    )
                            ),
                            useParallelTempering
                    );
                }
                case MarkovChainType::DikinWalk : {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            ModelMixin(
                                                    DikinProposal(inequalityLhs, inequalityRhs, startingPoint),
                                                    ColdnessAttribute(model)
                                            )
                                    )
                            ),
                            useParallelTempering
                    );
                }
                case MarkovChainType::HitAndRun: {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            ModelMixin(
                                                    HitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                            decltype(inequalityRhs),
                                                            GaussianStepDistribution<typename decltype(inequalityRhs)::Scalar>>(
                                                            inequalityLhs, inequalityRhs, startingPoint),
                                                    ColdnessAttribute(model)
                                            )
                                    )
                            ),
                            useParallelTempering
                    );
                }
                default: {
                    throw std::runtime_error("Type not supported for uniform sampling.");
                }
            }
        }

        /**
         * @brief Creats a Markov chain for sampling the likelihood of a model with the domain of a rounded convex polytope.
         * @tparam MatrixType
         * @tparam VectorType
         * @tparam Model
         * @param type
         * @param roundedInequalityLhs
         * @param roundedInequalityRhs
         * @param startingPoint
         * @param unroundingTransformation
         * @param unroundingShift
         * @param model
         * @return
         */
        template<typename MatrixType, typename VectorType, typename Model>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType roundedInequalityLhs,
                VectorType roundedInequalityRhs,
                VectorType startingPoint,
                MatrixType unroundingTransformation,
                VectorType unroundingShift,
                Model model,
                bool useParallelTempering
        ) {
            switch (type) {
                case MarkovChainType::BallWalk : {
                    return addRecordersAndAdapter(
                        MetropolisHastingsFilter(
                            ModelMixin(
                                StateTransformation(
                                    CoordinateHitAndRunProposal<
                                         Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                        decltype(roundedInequalityRhs),
                                        GaussianStepDistribution<typename decltype(roundedInequalityRhs)::Scalar>
                                    >(
                                        roundedInequalityLhs,
                                        roundedInequalityRhs,
                                        startingPoint
                                    ),
                                    Transformation(unroundingTransformation, unroundingShift)
                                ),
                                ColdnessAttribute(model)
                            )
                        ),
                        useParallelTempering
                    );
                }
                case MarkovChainType::CoordinateHitAndRun : {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            ModelMixin(
                                                    StateTransformation(
                                                            CoordinateHitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                    decltype(roundedInequalityRhs),
                                                                    GaussianStepDistribution<typename decltype(roundedInequalityRhs)::Scalar>>(
                                                                    roundedInequalityLhs,
                                                                    roundedInequalityRhs,
                                                                    startingPoint),
                                                            Transformation(unroundingTransformation, unroundingShift)
                                                    ),
                                                    ColdnessAttribute(model)
                                            )
                                    )
                            ),
                            useParallelTempering
                    );
                }
                case MarkovChainType::CSmMALA: {
                    throw std::runtime_error("CSmMALA not supported for rounded spaces. Use unrouded spaces.");
                }
                case MarkovChainType::CSmMALANoGradient: {
                    throw std::runtime_error("CSmMALANoGradient not supported for rounded spaces. Use unrouded spaces.");
                }
                case MarkovChainType::DikinWalk : {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            ModelMixin(
                                                    StateTransformation(
                                                            DikinProposal(
                                                                    roundedInequalityLhs,
                                                                    roundedInequalityRhs,
                                                                    startingPoint),
                                                            Transformation(unroundingTransformation, unroundingShift)
                                                    ),
                                                    ColdnessAttribute(model)
                                            )
                                    )
                            ),
                            useParallelTempering
                    );
                }
                case MarkovChainType::HitAndRun: {
                    return addRecordersAndAdapter(
                            NegativeLogLikelihoodRecorder(
                                    MetropolisHastingsFilter(
                                            ModelMixin(
                                                    StateTransformation(
                                                            HitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                    decltype(roundedInequalityRhs),
                                                                    GaussianStepDistribution<typename decltype(roundedInequalityRhs)::Scalar>>(
                                                                    roundedInequalityLhs,
                                                                    roundedInequalityRhs,
                                                                    startingPoint),
                                                            Transformation(unroundingTransformation, unroundingShift)
                                                    ),
                                                    ColdnessAttribute(model)
                                            )
                                    )
                            ),
                            useParallelTempering
                    );
                }
                default: {
                    throw std::runtime_error("Type not supported.");
                }
            }
        }


    private:
        /**
         * @brief Adds mixins to MarkovChain
         * @tparam MarkovChain
         * @return
         */
        template<typename MarkovChainImpl>
        static std::unique_ptr<MarkovChain>
        addRecordersAndAdapter(const MarkovChainImpl &markovChainImpl, bool useParallelTempering = false) {
            if (useParallelTempering) {
                if constexpr(IsSetColdnessAvailable<MarkovChainImpl>::value) {
                    auto mc = MarkovChainAdapter(
                            ParallelTempering(
                                    AcceptanceRateRecorder(
                                            TimestampRecorder(
                                                    StateRecorder(
                                                            markovChainImpl
                                                    )
                                            )
                                    )
                            )
                    );
                    return std::make_unique<decltype(mc)>(mc);
                } else {
                    throw std::runtime_error("Can not use Parallel Tempering without model.");
                }
            } else {
                auto mc = MarkovChainAdapter(
                        AcceptanceRateRecorder(
                                TimestampRecorder(
                                        StateRecorder(
                                                markovChainImpl
                                        )
                                )
                        )
                );
                return std::make_unique<decltype(mc)>(mc);
            }
        }
    };
}

#endif //HOPS_MARKOVCHAINFACTORY_HPP
