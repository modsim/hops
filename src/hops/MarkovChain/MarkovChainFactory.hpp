#ifndef HOPS_MARKOVCHAINFACTORY_HPP
#define HOPS_MARKOVCHAINFACTORY_HPP

#include <type_traits>

#include <hops/MarkovChain/Draw/NoOpDrawAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/ModelMixin.hpp>
#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainType.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/ParallelTempering/Coldness.hpp>
#include <hops/MarkovChain/ParallelTempering/ParallelTempering.hpp>
#include <hops/MarkovChain/Proposal/AdaptiveMetropolisProposal.hpp>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>
#include <hops/MarkovChain/Recorder/AcceptanceRateRecorder.hpp>
#include <hops/MarkovChain/Recorder/NegativeLogLikelihoodRecorder.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Recorder/TimestampRecorder.hpp>
#include <hops/MarkovChain/StateTransformation.hpp>
#include <hops/Model/Model.hpp>
#include <hops/Transformation/Transformation.hpp>
#include <utility>
#include "ModelWrapper.hpp"

namespace hops {
    class MarkovChainFactory {
    public:
        /**
         * @brief Creates a Markov chain for uniform sampling of convex polytopes using an arbitrary proposal class.
         * @tparam MatrixType
         * @tparam VectorType
         * @tparam Proposal
         * @param proposal
         * @return
         */
        template<typename MatrixType, typename VectorType, typename Proposal>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                Proposal proposal
        ) {
            return addRecordersAndAdapter(MetropolisHastingsFilter(proposal));
        }

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
            if (!isInteriorPoint(inequalityLhs, inequalityRhs, startingPoint)) {
                throw std::domain_error("Starting point outside polytope is always constant.");
            }

        }

        /**
         * @brief Creates a Markov chain for uniform sampling of rounded convex polytopes.
         * @tparam MatrixType
         * @tparam VectorType
         * @tparam Proposal
         * @param proposal
         * @param unroundingTransformation
         * @param unroundingShift
         * @return
         */
        template<typename MatrixType, typename VectorType, typename Proposal>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                Proposal proposal,
                MatrixType unroundingTransformation,
                VectorType unroundingShift
        ) {
            return addRecordersAndAdapter(
                    MetropolisHastingsFilter(
                            StateTransformation(
                                    proposal, Transformation(unroundingTransformation, unroundingShift)
                            )
                    )
            );
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
            if (!isInteriorPoint(roundedInequalityLhs, roundedInequalityRhs, startingPoint)) {
                throw std::domain_error("Starting point outside polytope is always constant.");
            }
        }

        /**
         * @brief Creates a Markov chain for sampling the likelihood of a with the domain of a convex polytope.
         * @tparam MatrixType
         * @tparam VectorType
         * @tparam Model
         * @tparam Proposal
         * @param proposal
         * @param model
         * @return
         */
        template<typename MatrixType, typename VectorType, typename Proposal>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                Proposal proposal,
                std::shared_ptr<Model> model
        ) {
            return addRecordersAndAdapter(
                    NegativeLogLikelihoodRecorder(
                            MetropolisHastingsFilter(
                                    ModelMixin(
                                            proposal,
                                            Coldness(ModelWrapper(std::move(model)))
                                    )
                            )
                    )
            );
        }

        /**
         * @brief Creates a Markov chain for sampling the likelihood of a model with the domain of a convex polytope.
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
        template<typename MatrixType, typename VectorType>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType inequalityLhs,
                VectorType inequalityRhs,
                VectorType startingPoint,
                const std::shared_ptr<Model> &model
        ) {
            if (!isInteriorPoint(inequalityLhs, inequalityRhs, startingPoint)) {
                throw std::domain_error("Starting point outside polytope is always constant.");
            }
        }

        /**
         * @brief Creates a Markov chain for sampling the likelihood of a model with the domain of a convex polytope with parallel tempering.
         * @tparam MatrixType
         * @tparam VectorType
         * @tparam Model
         * @param type
         * @param inequalityLhs
         * @param inequalityRhs
         * @param startingPoint
         * @param model
         * @param synchronizedRandomNumberGenerator required for efficient parallel tempering
         * @return
         */
        template<typename MatrixType, typename VectorType>
        static std::unique_ptr<MarkovChain> createMarkovChainWithParallelTempering(
                MarkovChainType type,
                MatrixType inequalityLhs,
                VectorType inequalityRhs,
                VectorType startingPoint,
                const std::shared_ptr<Model> &model,
                RandomNumberGenerator synchronizedRandomNumberGenerator
        ) {
            if (!isInteriorPoint(inequalityLhs, inequalityRhs, startingPoint)) {
                throw std::domain_error("Starting point outside polytope is always constant.");
            }
        }

        /**
         * @brief Creates a Markov chain for sampling the likelihood of a model with the domain of a rounded convex polytope.
         * @tparam MatrixType
         * @tparam VectorType
         * @tparam Model
         * @tparam Proposal
         * @param proposal
         * @param unroundingTransformation
         * @param unroundingShift
         * @param model
         * @return
         */
        template<typename MatrixType, typename VectorType, typename Proposal>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                Proposal proposal,
                MatrixType unroundingTransformation,
                VectorType unroundingShift,
                std::shared_ptr<Model> model
        ) {

            return addRecordersAndAdapter(
                    NegativeLogLikelihoodRecorder(
                            MetropolisHastingsFilter(
                                    ModelMixin(
                                            StateTransformation(
                                                    proposal,
                                                    Transformation(unroundingTransformation, unroundingShift)
                                            ),
                                            Coldness(ModelWrapper(model))
                                    )
                            )
                    )
            );
        }

        /**
         * @brief Creates a Markov chain for sampling the likelihood of a model with the domain of a rounded convex polytope.
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
        template<typename MatrixType, typename VectorType>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType roundedInequalityLhs,
                VectorType roundedInequalityRhs,
                VectorType startingPoint,
                MatrixType unroundingTransformation,
                VectorType unroundingShift,
                const std::shared_ptr<Model> &model
        ) {
            if (!isInteriorPoint(roundedInequalityLhs, roundedInequalityRhs, startingPoint)) {
                throw std::domain_error("Starting point outside polytope is always constant.");
            }
        }

        /**
         * @brief Creates a Markov chain for sampling the likelihood of a model with the domain of a rounded convex polytope with parallel tempering.
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
         * @param synchronizedRandomNumberGenerator (required for efficient parallel tempering)
         * @return
         */
        template<typename MatrixType, typename VectorType>
        static std::unique_ptr<MarkovChain> createMarkovChainWithParallelTempering(
                MarkovChainType type,
                MatrixType roundedInequalityLhs,
                VectorType roundedInequalityRhs,
                VectorType startingPoint,
                MatrixType unroundingTransformation,
                VectorType unroundingShift,
                const std::shared_ptr<Model> &model,
                RandomNumberGenerator synchronizedRandomNumberGenerator
        ) {
            if (!isInteriorPoint(roundedInequalityLhs, roundedInequalityRhs, startingPoint)) {
                throw std::domain_error("Starting point outside polytope is always constant.");
            }
        }


    private:
        template<typename MatrixType, typename VectorType>
        static bool isInteriorPoint(const MatrixType &A, const VectorType &b, const VectorType &x) {
            return ((b - A * x).array() >= 0).all();
        }

        /**
         * @brief Adds standard mixins to MarkovChain and sets up parallel tempering.
         * @tparam MarkovChain
         * @return
         */
        template<typename MarkovChainImpl>
        static std::unique_ptr<MarkovChain>
        addRecordersAndAdapter(const MarkovChainImpl &markovChainImpl,
                               RandomNumberGenerator synchronizedRandomNumberGenerator) {
            if constexpr(IsSetColdnessAvailable<MarkovChainImpl>::value) {
                auto mc = MarkovChainAdapter(
                        ParallelTempering(
                                AcceptanceRateRecorder(
                                        TimestampRecorder(
                                                StateRecorder(
                                                        markovChainImpl
                                                )
                                        )
                                ),
                                synchronizedRandomNumberGenerator
                        )
                );
                return std::make_unique<decltype(mc)>(mc);
            } else {
                throw std::invalid_argument("Can not use Parallel Tempering without model.");
            }
        };

        /**
         * @brief Adds standard mixins to MarkovChain
         * @tparam MarkovChain
         * @return
         */
        template<typename MarkovChainImpl>
        static std::unique_ptr<MarkovChain>
        addRecordersAndAdapter(const MarkovChainImpl &markovChainImpl) {
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
    };
}

#endif //HOPS_MARKOVCHAINFACTORY_HPP
