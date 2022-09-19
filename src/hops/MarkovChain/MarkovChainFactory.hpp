#ifndef HOPS_MARKOVCHAINFACTORY_HPP
#define HOPS_MARKOVCHAINFACTORY_HPP

#include <type_traits>

#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/MarkovChainType.hpp"
#include "hops/MarkovChain/MarkovChainAdapter.hpp"
#include "hops/MarkovChain/Draw/NoOpDrawAdapter.hpp"
#include "hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp"
#include "hops/MarkovChain/ParallelTempering/Coldness.hpp"
#include "hops/MarkovChain/ParallelTempering/ParallelTempering.hpp"
#include "hops/MarkovChain/Proposal/AdaptiveMetropolisProposal.hpp"
#include "hops/MarkovChain/Proposal/BallWalkProposal.hpp"
#include "hops/MarkovChain/Proposal/BilliardAdaptiveMetropolisProposal.hpp"
#include "hops/MarkovChain/Proposal/BilliardMALAProposal.hpp"
#include "hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp"
#include "hops/MarkovChain/Proposal/CSmMALAProposal.hpp"
#include "hops/MarkovChain/Proposal/DikinProposal.hpp"
#include "hops/MarkovChain/Proposal/GaussianProposal.hpp"
#include "hops/MarkovChain/Proposal/HitAndRunProposal.hpp"
#include "hops/MarkovChain/Recorder/AcceptanceRateRecorder.hpp"
#include "hops/MarkovChain/Recorder/NegativeLogLikelihoodRecorder.hpp"
#include "hops/MarkovChain/Recorder/StateRecorder.hpp"
#include "hops/MarkovChain/Recorder/TimestampRecorder.hpp"
#include "hops/MarkovChain/StateTransformation.hpp"
#include "hops/MarkovChain/ModelMixin.hpp"
#include "hops/MarkovChain/ModelWrapper.hpp"
#include "hops/Transformation/LinearTransformation.hpp"
#include "hops/MarkovChain/Recorder/NegativeLogLikelihoodRecorder.hpp"

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
        );

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
        );

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
        );

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
        );

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
        template<typename MatrixType, typename VectorType, typename Model, typename Proposal>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                Proposal proposal,
                Model model
        );

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
        template<typename MatrixType, typename VectorType, typename Model>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType inequalityLhs,
                VectorType inequalityRhs,
                VectorType startingPoint,
                Model model
        );

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
        template<typename MatrixType, typename VectorType, typename Model>
        static std::unique_ptr<MarkovChain> createMarkovChainWithParallelTempering(
                MarkovChainType type,
                MatrixType inequalityLhs,
                VectorType inequalityRhs,
                VectorType startingPoint,
                Model model,
                RandomNumberGenerator synchronizedRandomNumberGenerator
        );

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
        template<typename MatrixType, typename VectorType, typename Model, typename Proposal>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                Proposal proposal,
                MatrixType unroundingTransformation,
                VectorType unroundingShift,
                Model model
        );

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
        template<typename MatrixType, typename VectorType, typename Model>
        static std::unique_ptr<MarkovChain> createMarkovChain(
                MarkovChainType type,
                MatrixType roundedInequalityLhs,
                VectorType roundedInequalityRhs,
                VectorType startingPoint,
                MatrixType unroundingTransformation,
                VectorType unroundingShift,
                Model model
        );

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
        template<typename MatrixType, typename VectorType, typename Model>
        static std::unique_ptr<MarkovChain> createMarkovChainWithParallelTempering(
                MarkovChainType type,
                MatrixType roundedInequalityLhs,
                VectorType roundedInequalityRhs,
                VectorType startingPoint,
                MatrixType unroundingTransformation,
                VectorType unroundingShift,
                Model model,
                RandomNumberGenerator synchronizedRandomNumberGenerator
        );

    private:
        template<typename MatrixType, typename VectorType>
        static bool isInteriorPoint(const MatrixType &A, const VectorType &b, const VectorType &x) {
            return ((b - A * x).array() >= 0).all();
        }
    };


    template<typename MatrixType, typename VectorType, typename Proposal>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            Proposal proposal
    ) {
        return wrapMarkovChainImpl(MarkovChainAdapter(MetropolisHastingsFilter(proposal)));
    }


    template<typename MatrixType, typename VectorType>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            MarkovChainType type,
            MatrixType inequalityLhs,
            VectorType inequalityRhs,
            VectorType startingPoint
    ) {
        if (!isInteriorPoint(inequalityLhs, inequalityRhs, startingPoint)) {
            throw std::runtime_error("Starting point outside polytope is always constant.");
        }

        switch (type) {
            case MarkovChainType::BallWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                NoOpDrawAdapter(
                                        BallWalkProposal(
                                                Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                        inequalityLhs), inequalityRhs, startingPoint)
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRun : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                NoOpDrawAdapter(
                                        CoordinateHitAndRunProposal(
                                                Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                        inequalityLhs), inequalityRhs, startingPoint)
                                )
                        )
                );
            }
            case MarkovChainType::DikinWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        DikinProposal(inequalityLhs, inequalityRhs, startingPoint)
                                )
                        )
                );
            }
            case MarkovChainType::Gaussian : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        GaussianProposal(
                                                Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                        inequalityLhs), inequalityRhs, startingPoint)
                                )
                        )
                );
            }
            case MarkovChainType::HitAndRun: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                NoOpDrawAdapter(
                                        HitAndRunProposal(
                                                Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                        inequalityLhs), inequalityRhs, startingPoint)
                                )
                        )
                );
            }
            default: {
                throw std::runtime_error("MarkovChainType not supported for uniform sampling.");
            }
        }
    }


    template<typename MatrixType, typename VectorType, typename Proposal>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            Proposal proposal,
            MatrixType unroundingTransformation,
            VectorType unroundingShift
    ) {
        return wrapMarkovChainImpl(
                MarkovChainAdapter(
                        MetropolisHastingsFilter(
                                StateTransformation(
                                        proposal, LinearTransformation(unroundingTransformation, unroundingShift)
                                )
                        )
                )
        );
    }


    template<typename MatrixType, typename VectorType>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            MarkovChainType type,
            MatrixType roundedInequalityLhs,
            VectorType roundedInequalityRhs,
            VectorType startingPoint,
            MatrixType unroundingTransformation,
            VectorType unroundingShift
    ) {
        if (!isInteriorPoint(roundedInequalityLhs, roundedInequalityRhs, startingPoint)) {
            throw std::runtime_error("Starting point outside polytope is always constant.");
        }

        switch (type) {
            case MarkovChainType::BallWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                NoOpDrawAdapter(
                                        StateTransformation(
                                                BallWalkProposal(
                                                        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                                roundedInequalityLhs),
                                                        roundedInequalityRhs,
                                                        startingPoint),
                                                LinearTransformation(unroundingTransformation, unroundingShift)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRun : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                NoOpDrawAdapter(
                                        StateTransformation(
                                                CoordinateHitAndRunProposal(
                                                        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                                roundedInequalityLhs),
                                                        roundedInequalityRhs,
                                                        startingPoint),
                                                LinearTransformation(unroundingTransformation, unroundingShift)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::DikinWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        StateTransformation(
                                                DikinProposal(
                                                        roundedInequalityLhs,
                                                        roundedInequalityRhs,
                                                        startingPoint),
                                                LinearTransformation(unroundingTransformation, unroundingShift)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::Gaussian : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        StateTransformation(
                                                GaussianProposal(
                                                        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                                roundedInequalityLhs),
                                                        roundedInequalityRhs,
                                                        startingPoint),
                                                LinearTransformation(unroundingTransformation, unroundingShift)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::HitAndRun: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                NoOpDrawAdapter(
                                        StateTransformation(
                                                HitAndRunProposal(
                                                        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>(
                                                                roundedInequalityLhs),
                                                        roundedInequalityRhs,
                                                        startingPoint),
                                                LinearTransformation(unroundingTransformation, unroundingShift)
                                        )
                                )
                        )
                );
            }
            default: {
                throw std::runtime_error("MarkovChainType not supported for uniform sampling.");
            }
        }
    }


    template<typename MatrixType, typename VectorType, typename Model, typename Proposal>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            Proposal proposal,
            Model model
    ) {
        return wrapMarkovChainImpl(
                MarkovChainAdapter(
                        MetropolisHastingsFilter(
                                ModelMixin(
                                        proposal,
                                        Coldness(model)
                                )
                        )
                )
        );
    }

    template<typename MatrixType, typename VectorType, typename Model>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            MarkovChainType type,
            MatrixType inequalityLhs,
            VectorType inequalityRhs,
            VectorType startingPoint,
            Model model
    ) {
        if (!isInteriorPoint(inequalityLhs, inequalityRhs, startingPoint)) {
            throw std::runtime_error("Starting point outside polytope is always constant.");
        }
        switch (type) {
            case MarkovChainType::AdaptiveMetropolis : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                AdaptiveMetropolisProposal<
                                                        Eigen::Matrix<
                                                                typename MatrixType::Scalar,
                                                                Eigen::Dynamic,
                                                                Eigen::Dynamic>>(inequalityLhs,
                                                                                 inequalityRhs,
                                                                                 startingPoint),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::BallWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                BallWalkProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                        decltype(inequalityRhs)>(
                                                        inequalityLhs,
                                                        inequalityRhs,
                                                        startingPoint),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::BilliardAdaptiveMetropolis : {
                // estimated from https://arxiv.org/pdf/2102.13068.pdf
                long maxNumberOfReflections = inequalityLhs.cols() * 100;
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                BilliardAdaptiveMetropolisProposal(
                                                        AdaptiveMetropolisProposal<
                                                                Eigen::Matrix<
                                                                        typename MatrixType::Scalar,
                                                                        Eigen::Dynamic,
                                                                        Eigen::Dynamic>>(
                                                                inequalityLhs,
                                                                inequalityRhs,
                                                                startingPoint),
                                                        maxNumberOfReflections
                                                ),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::BilliardMALA : {
                // estimated from https://arxiv.org/pdf/2102.13068.pdf
                long maxNumberOfReflections = inequalityLhs.cols() * 100;
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        BilliardMALAProposal(
                                                inequalityLhs,
                                                inequalityRhs,
                                                startingPoint,
                                                Coldness(model),
                                                maxNumberOfReflections
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRun : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                CoordinateHitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                        decltype(inequalityRhs),
                                                        GaussianStepDistribution<typename decltype(inequalityRhs)::Scalar>>(
                                                        inequalityLhs,
                                                        inequalityRhs,
                                                        startingPoint),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::CSmMALA: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        CSmMALAProposal(
                                                inequalityLhs,
                                                inequalityRhs,
                                                startingPoint,
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::DikinWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                DikinProposal(inequalityLhs, inequalityRhs, startingPoint),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::Gaussian : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                GaussianProposal<
                                                        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                        decltype(inequalityRhs)
                                                >(inequalityLhs, inequalityRhs, startingPoint),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::HitAndRun: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                HitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                        decltype(inequalityRhs),
                                                        GaussianStepDistribution<typename decltype(inequalityRhs)::Scalar>>(
                                                        inequalityLhs, inequalityRhs, startingPoint),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            default: {
                throw std::runtime_error("Type not supported.");
            }
        }
    }

    template<typename MatrixType, typename VectorType, typename Model>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChainWithParallelTempering(
            MarkovChainType type,
            MatrixType inequalityLhs,
            VectorType inequalityRhs,
            VectorType startingPoint,
            Model model,
            RandomNumberGenerator synchronizedRandomNumberGenerator
    ) {
        if (!isInteriorPoint(inequalityLhs, inequalityRhs, startingPoint)) {
            throw std::runtime_error("Starting point outside polytope is always constant.");
        }

        switch (type) {
            case MarkovChainType::AdaptiveMetropolis : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        AdaptiveMetropolisProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>>(
                                                                inequalityLhs, inequalityRhs, startingPoint),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::BallWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        BallWalkProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(inequalityRhs)>(
                                                                inequalityLhs, inequalityRhs, startingPoint),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRun : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        CoordinateHitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(inequalityRhs),
                                                                GaussianStepDistribution<typename decltype(inequalityRhs)::Scalar>>(
                                                                inequalityLhs, inequalityRhs, startingPoint),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::CSmMALA: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                CSmMALAProposal(inequalityLhs,
                                                                inequalityRhs,
                                                                startingPoint,
                                                                Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::DikinWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        DikinProposal(inequalityLhs, inequalityRhs, startingPoint),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::Gaussian : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        GaussianProposal<
                                                                Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(inequalityRhs)
                                                        >(inequalityLhs, inequalityRhs, startingPoint),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::HitAndRun: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        HitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(inequalityRhs),
                                                                GaussianStepDistribution<typename decltype(inequalityRhs)::Scalar>>(
                                                                inequalityLhs, inequalityRhs, startingPoint),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            default: {
                throw std::runtime_error("Type not supported.");
            }
        }
    }


    template<typename MatrixType, typename VectorType, typename Model, typename Proposal>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            Proposal proposal,
            MatrixType unroundingTransformation,
            VectorType unroundingShift,
            Model model
    ) {
        return wrapMarkovChainImpl(
                MarkovChainAdapter(
                        MetropolisHastingsFilter(
                                ModelMixin(
                                        StateTransformation(
                                                proposal,
                                                LinearTransformation(unroundingTransformation, unroundingShift)
                                        ),
                                        model
                                )
                        )
                )
        );
    }


    template<typename MatrixType, typename VectorType, typename Model>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChain(
            MarkovChainType type,
            MatrixType roundedInequalityLhs,
            VectorType roundedInequalityRhs,
            VectorType startingPoint,
            MatrixType unroundingTransformation,
            VectorType unroundingShift,
            Model model
    ) {
        if (!isInteriorPoint(roundedInequalityLhs, roundedInequalityRhs, startingPoint)) {
            throw std::runtime_error("Starting point outside polytope is always constant.");
        }
        switch (type) {
            case MarkovChainType::AdaptiveMetropolis : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                StateTransformation(
                                                        AdaptiveMetropolisProposal(
                                                                roundedInequalityLhs, roundedInequalityRhs,
                                                                startingPoint,
                                                                // MaxVolEllipsoid is identity because we are in rounded space
                                                                decltype(roundedInequalityLhs)::Identity(
                                                                        roundedInequalityLhs.cols(),
                                                                        roundedInequalityLhs.cols())),
                                                        LinearTransformation(unroundingTransformation,
                                                                             unroundingShift)
                                                ),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::BallWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                StateTransformation(
                                                        BallWalkProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(roundedInequalityRhs)>(
                                                                roundedInequalityLhs,
                                                                roundedInequalityRhs,
                                                                startingPoint),
                                                        LinearTransformation(unroundingTransformation,
                                                                             unroundingShift)
                                                ),
                                                model
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::BilliardAdaptiveMetropolis : {
                long maxReflections =
                        roundedInequalityLhs.cols() * 100; // estimated from https://arxiv.org/pdf/2102.13068.pdf
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                StateTransformation(
                                                        BilliardAdaptiveMetropolisProposal(
                                                                AdaptiveMetropolisProposal(
                                                                        roundedInequalityLhs, roundedInequalityRhs,
                                                                        startingPoint,
                                                                        // MaxVolEllipsoid is identity because we are in rounded space
                                                                        decltype(roundedInequalityLhs)::Identity(
                                                                                roundedInequalityLhs.cols(),
                                                                                roundedInequalityLhs.cols())),
                                                                maxReflections
                                                        ), LinearTransformation(unroundingTransformation,
                                                                                unroundingShift)
                                                ),
                                                Coldness(model)
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRun : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                StateTransformation(
                                                        CoordinateHitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(roundedInequalityRhs),
                                                                GaussianStepDistribution<typename decltype(roundedInequalityRhs)::Scalar>>(
                                                                roundedInequalityLhs,
                                                                roundedInequalityRhs,
                                                                startingPoint),
                                                        LinearTransformation(unroundingTransformation,
                                                                             unroundingShift)
                                                ),
                                                model
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::Gaussian : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                StateTransformation(
                                                        GaussianProposal<
                                                                Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(roundedInequalityRhs)>(
                                                                roundedInequalityLhs,
                                                                roundedInequalityRhs,
                                                                startingPoint
                                                        ),
                                                        LinearTransformation(unroundingTransformation,
                                                                             unroundingShift)
                                                ),
                                                model
                                        )
                                )
                        )
                );
            }
            case MarkovChainType::HitAndRun: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                MetropolisHastingsFilter(
                                        ModelMixin(
                                                StateTransformation(
                                                        HitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                decltype(roundedInequalityRhs),
                                                                GaussianStepDistribution<typename decltype(roundedInequalityRhs)::Scalar>>(
                                                                roundedInequalityLhs,
                                                                roundedInequalityRhs,
                                                                startingPoint),
                                                        LinearTransformation(unroundingTransformation,
                                                                             unroundingShift)
                                                ),
                                                model
                                        )
                                )
                        )
                );
            }
            default: {
                throw std::runtime_error("Type not supported.");
            }
        }
    }


    template<typename MatrixType, typename VectorType, typename Model>
    std::unique_ptr<MarkovChain> MarkovChainFactory::createMarkovChainWithParallelTempering(
            MarkovChainType type,
            MatrixType roundedInequalityLhs,
            VectorType roundedInequalityRhs,
            VectorType startingPoint,
            MatrixType unroundingTransformation,
            VectorType unroundingShift,
            Model model,
            RandomNumberGenerator synchronizedRandomNumberGenerator
    ) {
        if (!isInteriorPoint(roundedInequalityLhs, roundedInequalityRhs, startingPoint)) {
            throw std::runtime_error("Starting point outside polytope is always constant.");
        }
        // TODO add AAM,
        // TODO add BAM,
        // TODO add BilliardWalk
        switch (type) {
            case MarkovChainType::BallWalk : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        StateTransformation(
                                                                BallWalkProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                        decltype(roundedInequalityRhs)>(
                                                                        roundedInequalityLhs,
                                                                        roundedInequalityRhs,
                                                                        startingPoint),
                                                                LinearTransformation(unroundingTransformation,
                                                                                     unroundingShift)
                                                        ),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::CoordinateHitAndRun : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        StateTransformation(
                                                                CoordinateHitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                        decltype(roundedInequalityRhs),
                                                                        GaussianStepDistribution<typename decltype(roundedInequalityRhs)::Scalar>>(
                                                                        roundedInequalityLhs,
                                                                        roundedInequalityRhs,
                                                                        startingPoint),
                                                                LinearTransformation(unroundingTransformation,
                                                                                     unroundingShift)
                                                        ),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::Gaussian : {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        StateTransformation(
                                                                GaussianProposal<
                                                                        Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                        decltype(roundedInequalityRhs)>(
                                                                        roundedInequalityLhs,
                                                                        roundedInequalityRhs,
                                                                        startingPoint
                                                                ),
                                                                LinearTransformation(unroundingTransformation,
                                                                                     unroundingShift)
                                                        ),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            case MarkovChainType::HitAndRun: {
                return wrapMarkovChainImpl(
                        MarkovChainAdapter(
                                ParallelTempering(
                                        MetropolisHastingsFilter(
                                                ModelMixin(
                                                        StateTransformation(
                                                                HitAndRunProposal<Eigen::Matrix<typename MatrixType::Scalar, Eigen::Dynamic, Eigen::Dynamic>,
                                                                        decltype(roundedInequalityRhs),
                                                                        GaussianStepDistribution<typename decltype(roundedInequalityRhs)::Scalar>>(
                                                                        roundedInequalityLhs,
                                                                        roundedInequalityRhs,
                                                                        startingPoint),
                                                                LinearTransformation(unroundingTransformation,
                                                                                     unroundingShift)
                                                        ),
                                                        Coldness(model)
                                                )
                                        ),
                                        synchronizedRandomNumberGenerator
                                )
                        )
                );
            }
            default: {
                throw std::runtime_error("Type not supported.");
            }
        }
    }

    template<typename MarkovChainImpl>
    std::unique_ptr<MarkovChain> wrapMarkovChainImpl(const MarkovChainImpl &markovChain) {
        return std::make_unique<MarkovChainImpl>(markovChain);
    }


}

#endif //HOPS_MARKOVCHAINFACTORY_HPP
