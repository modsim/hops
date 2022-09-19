#ifndef HOPS_SAMPLING_HPP
#define HOPS_SAMPLING_HPP

#include <chrono>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <iomanip>
#include <utility>

#include "hops/MarkovChain/MarkovChainFactory.hpp"
#include "hops/FileWriter/FileWriter.hpp"
#include "hops/FileWriter/FileWriterType.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/MarkovChainType.hpp"
#include "hops/Polytope/MaximumVolumeEllipsoid.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {
    class Sampling {
    public:
        static bool tuneChain(hops::RandomNumberGenerator &randomNumberGenerator,
                              double targetAcceptanceRate,
                              std::shared_ptr<hops::MarkovChain> markovChain,
                              hops::MarkovChainType chainType,
                              hops::FileWriter *fileWriter,
                              double tuningTolerance,
                              long maxTuningIterations);

        template<typename ModelType>
        static void run(const MatrixType &A,
                        const VectorType &b,
                        const VectorType &startPoint,
                        const ModelType &model,
                        hops::MarkovChainType chainType,
                        long numberOfSamples,
                        long thinning,
                        long numberOfCheckpoints,
                        double fisherWeight,
                        double targetAcceptanceRate,
                        bool rounding,
                        const std::string &problemName,
                        double startStepsize = 1,
                        bool tune = true,
                        double tuningTolerance = 0.05,
                        long maxTuningIterations = 10);

        /**
         *
         * @tparam ModelType
         * @param A
         * @param b
         * @param startPoint
         * @param roundingTransformation  IMPORTANT: Should be lower triangular
         * @param shift
         * @param model
         * @param chainType
         * @param numberOfSamples
         * @param fisherWeight
         * @param targetAcceptanceRate
         * @param rounding
         * @param problemName
         */
        template<typename ModelType>
        static void run(const MatrixType &A,
                        const VectorType &b,
                        const VectorType &startPoint,
                        const MatrixType &roundingTransformation,
                        const MatrixType &shift,
                        const ModelType &model,
                        hops::MarkovChainType chainType,
                        long numberOfSamples,
                        long thinning,
                        long numberOfCheckpoints,
                        double fisherWeight,
                        double targetAcceptanceRate,
                        bool rounding,
                        const std::string &problemName,
                        double startStepsize = 1,
                        bool tune = true,
                        double tuningTolerance = 0.05,
                        long maxTuningIterations = 10);
    };

    template<typename ModelType>
    void Sampling::run(const MatrixType &A,
                       const VectorType &b,
                       const VectorType &startPoint,
                       const ModelType &model,
                       MarkovChainType chainType,
                       long numberOfSamples,
                       long thinning,
                       long numberOfCheckpoints,
                       double fisherWeight,
                       double targetAcceptanceRate,
                       bool rounding,
                       const std::string &problemName,
                       double startStepsize,
                       bool tune,
                       double tuningTolerance,
                       long maxTuningIterations) {

        Eigen::MatrixXd roundingTrafo = Eigen::MatrixXd::Identity(startPoint.rows(), startPoint.rows());
        Eigen::VectorXd shift = Eigen::VectorXd::Zero(startPoint.rows());
        Eigen::VectorXd startPointRounded = startPoint;

        if (rounding) {
            auto MVE = MaximumVolumeEllipsoid<double>::construct(A, b, 100000);
            if (!MVE.getRoundingTransformation().isLowerTriangular()) {
                throw std::runtime_error(
                        "Expected hops::MaximumVolumeEllipsoid to construct lower triangular rounding transformation, but got something else.");
            }
            startPointRounded = MVE.getRoundingTransformation().template triangularView<Eigen::Lower>().
                    solve(startPoint);
            roundingTrafo = MVE.getRoundingTransformation();
        }
        run(A,
            b,
            startPointRounded,
            roundingTrafo,
            shift,
            model,
            chainType,
            numberOfSamples,
            thinning,
            numberOfCheckpoints,
            fisherWeight,
            targetAcceptanceRate,
            rounding,
            problemName,
            startStepsize,
            tune,
            tuningTolerance,
            maxTuningIterations);
    }

    template<typename ModelType>
    void Sampling::run(const MatrixType &A,
                       const VectorType &b,
                       const VectorType &startPoint,
                       const MatrixType &roundingTransformation,
                       const MatrixType &shift,
                       const ModelType &model,
                       MarkovChainType chainType,
                       long numberOfSamples,
                       long thinning,
                       long numberOfCheckPoints,
                       double fisherWeight,
                       double targetAcceptanceRate,
                       bool rounding,
                       const std::string &problemName,
                       double startStepsize,
                       bool tune,
                       double tuningTolerance,
                       long maxTuningIterations) {

        if (numberOfCheckPoints >= 10 * numberOfSamples) {
            throw std::invalid_argument(
                    "number of checkpoints is too high for number of samples selected. Aim for at least less than one checkpoint for every 10 samples!");
        }

        std::shared_ptr<MarkovChain> markovChain;
        if (rounding) {
            Eigen::VectorXd startPointRounded;
            if (roundingTransformation.isLowerTriangular()) {
                startPointRounded = roundingTransformation.template triangularView<Eigen::Lower>().
                        solve(startPoint);
            } else if (roundingTransformation.isUpperTriangular()) {
                startPointRounded = roundingTransformation.template triangularView<Eigen::Upper>().
                        solve(startPoint);
            }
            Eigen::MatrixXd Arounded = A * roundingTransformation;
            Eigen::MatrixXd unroundingMatrix = roundingTransformation;

            markovChain = MarkovChainFactory::createMarkovChain(
                    chainType,
                    Arounded,
                    b,
                    startPointRounded,
                    unroundingMatrix,
                    Eigen::VectorXd(Eigen::VectorXd::Zero(unroundingMatrix.cols())),
                    model);

        } else {
            Eigen::SparseMatrix<typename MatrixType::Scalar> sparseA = A.sparseView();
            bool isSparse = sparseA.nonZeros() < 1. / 2 * A.cols() * A.rows();
            if (isSparse) {
                markovChain = MarkovChainFactory::createMarkovChain(
                        chainType,
                        sparseA,
                        b,
                        startPoint,
                        model);
            } else {
                markovChain = MarkovChainFactory::createMarkovChain(
                        chainType,
                        A,
                        b,
                        startPoint,
                        model);
            }
        }

        if (chainType == MarkovChainType::CSmMALA) {
            markovChain->setParameter(ProposalParameter::FISHER_WEIGHT, fisherWeight);
        }
        markovChain->setParameter(ProposalParameter::STEP_SIZE, startStepsize);


        RandomNumberGenerator randomNumberGenerator((std::random_device()()));


        std::vector<double> acceptanceRates;
        std::vector<double> negLogLikelihoods;
        std::vector<Eigen::VectorXd> states;
        std::vector<long> timestamps;

        acceptanceRates.reserve(numberOfSamples);
        negLogLikelihoods.reserve(numberOfSamples);
        states.reserve(numberOfSamples);
        timestamps.reserve(numberOfSamples);


        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << fisherWeight;
        std::string fisherWeightString = stream.str();

        std::unique_ptr<FileWriter> writer = FileWriterFactory::createFileWriter(
                problemName + "_" +
                markovChainTypeToShortString(chainType)
                + (chainType == MarkovChainType::CSmMALA ? "_fw=" + fisherWeightString : "")
                + (rounding ? "_rounded" : ""), FileWriterType::CSV);

        if (tune) {
            bool isTuned = false;
            isTuned = Sampling::tuneChain(randomNumberGenerator, targetAcceptanceRate, markovChain, chainType,
                                          writer.get(), tuningTolerance, maxTuningIterations);
            writer->write("tuning_successful", std::vector<double>{static_cast<double>(isTuned)});
        } else {
            writer->write("tuning_successful", std::vector<std::string>{"skipped"});
        }

        for (long checkPoint = 0; checkPoint < numberOfCheckPoints; ++checkPoint) {
            for (long i = 0; i < numberOfSamples / numberOfCheckPoints; ++i) {
                ABORTABLE
                auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator,
                                                                                           thinning);
                acceptanceRates.emplace_back(acceptanceRate);
                negLogLikelihoods.emplace_back(markovChain->getStateNegativeLogLikelihood());
                states.emplace_back(state);
                timestamps.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()
                ).count());
            }

            writer->write("states", states);
            writer->write("acceptance_rates", std::vector<double>{
                    std::reduce(acceptanceRates.begin(), acceptanceRates.end(), 0.) / acceptanceRates.size()});
            writer->write("neg_log_likelihoods", negLogLikelihoods);
            writer->write("timestamps", timestamps);

            writer->write("parameter_stepSize", std::vector<double>{
                    std::any_cast<double>(markovChain->getParameter(ProposalParameter::STEP_SIZE))});
            writer->write("parameter_acceptance_rate_target", std::vector<double>{targetAcceptanceRate});
            states.clear();
            acceptanceRates.clear();
            negLogLikelihoods.clear();
            timestamps.clear();
        }
    }
}


#endif //HOPS_SAMPLING_HPP
