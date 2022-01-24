#ifndef HOPS_SAMPLING_HPP
#define HOPS_SAMPLING_HPP

#include <chrono>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <iomanip>
#include <utility>

#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include "hops/FileWriter/FileWriter.hpp"
#include "hops/FileWriter/FileWriterType.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/MarkovChain/MarkovChain.hpp"
#include "hops/MarkovChain/MarkovChainType.hpp"
#include <hops/Polytope/MaximumVolumeEllipsoid.hpp>
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
                              double tuningTolerance = 0.03);

        template<typename ModelType>
        static void run(const MatrixType &A,
                        const VectorType &b,
                        const VectorType &startPoint,
                        const ModelType &model,
                        hops::MarkovChainType chainType,
                        int numberOfSamples,
                        double fisherWeight,
                        double targetAcceptanceRate,
                        bool rounding,
                        const std::string &problemName);

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
                        int numberOfSamples,
                        double fisherWeight,
                        double targetAcceptanceRate,
                        bool rounding,
                        const std::string &problemName);
    };

    template<typename ModelType>
    void Sampling::run(const MatrixType &A,
                             const VectorType &b,
                             const VectorType &startPoint,
                             const ModelType &model,
                             MarkovChainType chainType,
                             int numberOfSamples,
                             double fisherWeight,
                             double targetAcceptanceRate,
                             bool rounding,
                             const std::string &problemName) {
        std::shared_ptr<MarkovChain> markovChain;

        if (rounding) {
            auto MVE = MaximumVolumeEllipsoid<double>::construct(A, b, 100000);
            Eigen::VectorXd startPointRounded = MVE.getRoundingTransformation().template triangularView<Eigen::Lower>().
                    solve(startPoint);
            Eigen::MatrixXd Arounded = A * MVE.getRoundingTransformation();
            Eigen::MatrixXd unroundingMatrix = MVE.getRoundingTransformation();

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


        RandomNumberGenerator randomNumberGenerator((std::random_device()()));


        std::vector<double> acceptanceRates;
        std::vector<double> negLogLikelihoods;
        std::vector<Eigen::VectorXd> states;
        std::vector<long> timestamps;
        std::vector<ProposalStatistics> proposalStatistics;

        acceptanceRates.reserve(numberOfSamples);
        negLogLikelihoods.reserve(numberOfSamples);
        states.reserve(numberOfSamples);
        timestamps.reserve(numberOfSamples);


        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << fisherWeight;
        std::string fisherWeightString = stream.str();

        std::unique_ptr<FileWriter> writer = FileWriterFactory::createFileWriter(
                problemName + std::to_string(A.cols()) + "_" +
                markovChainTypeToShortString(chainType)
                + (chainType == MarkovChainType::CSmMALA ? "_fw=" + fisherWeightString : "")
                + (rounding ? "_rounded" : ""), FileWriterType::CSV);

        bool isTuned = false;
        if (chainType != MarkovChainType::BilliardMALA &&
            chainType != MarkovChainType::BilliardAdaptiveMetropolis &&
            chainType != MarkovChainType::AdaptiveMetropolis) {
            isTuned = Sampling::tuneChain(randomNumberGenerator, targetAcceptanceRate, markovChain, chainType,
                                                writer.get());
        }
        writer->write("tuning_successful", std::vector<double>{static_cast<double>(isTuned)});

        for (int i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state, proposalStatistic] = markovChain->detailedDraw(randomNumberGenerator, 1);
            acceptanceRates.emplace_back(acceptanceRate);
            negLogLikelihoods.emplace_back(markovChain->getStateNegativeLogLikelihood());
            states.emplace_back(state);
            timestamps.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count());
            proposalStatistics.template emplace_back(proposalStatistic);
        }


        for (auto &statistic : proposalStatistics) {
            auto statisticMap = statistic.getStatistics();
            for (auto &sm: statisticMap) {
                writer->write(sm.first, sm.second);
            }
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

    template<typename ModelType>
    void Sampling::run(const MatrixType &A, const VectorType &b, const VectorType &startPoint,
                             const MatrixType &roundingTransformation, const MatrixType &shift,
                             const ModelType &model, MarkovChainType chainType, int numberOfSamples,
                             double fisherWeight, double targetAcceptanceRate, bool rounding,
                             const std::string &problemName) {
        std::shared_ptr<MarkovChain> markovChain;
        if (rounding) {
            Eigen::VectorXd startPointRounded = roundingTransformation.template triangularView<Eigen::Lower>().
                    solve(startPoint);
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


        RandomNumberGenerator randomNumberGenerator((std::random_device()()));


        std::vector<double> acceptanceRates;
        std::vector<double> negLogLikelihoods;
        std::vector<Eigen::VectorXd> states;
        std::vector<long> timestamps;
        std::vector<ProposalStatistics> proposalStatistics;

        acceptanceRates.reserve(numberOfSamples);
        negLogLikelihoods.reserve(numberOfSamples);
        states.reserve(numberOfSamples);
        timestamps.reserve(numberOfSamples);


        std::stringstream stream;
        stream << std::fixed << std::setprecision(1) << fisherWeight;
        std::string fisherWeightString = stream.str();

        std::unique_ptr<FileWriter> writer = FileWriterFactory::createFileWriter(
                problemName + std::to_string(A.cols()) + "_" +
                markovChainTypeToShortString(chainType)
                + (chainType == MarkovChainType::CSmMALA ? "_fw=" + fisherWeightString : "")
                + (rounding ? "_rounded" : ""), FileWriterType::CSV);

        bool isTuned = false;
        if (chainType != MarkovChainType::BilliardMALA &&
            chainType != MarkovChainType::BilliardAdaptiveMetropolis &&
            chainType != MarkovChainType::AdaptiveMetropolis) {
            isTuned = Sampling::tuneChain(randomNumberGenerator, targetAcceptanceRate, markovChain, chainType,
                                                writer.get());
        }
        writer->write("tuning_successful", std::vector<double>{static_cast<double>(isTuned)});

        for (int i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state, proposalStatistic] = markovChain->detailedDraw(randomNumberGenerator, 1);
            acceptanceRates.emplace_back(acceptanceRate);
            negLogLikelihoods.emplace_back(markovChain->getStateNegativeLogLikelihood());
            states.emplace_back(state);
            timestamps.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count());
            proposalStatistics.template emplace_back(proposalStatistic);
        }


        for (auto &statistic : proposalStatistics) {
            auto statisticMap = statistic.getStatistics();
            for (auto &sm: statisticMap) {
                writer->write(sm.first, sm.second);
            }
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


#endif //HOPS_SAMPLING_HPP
