#ifndef HOPS_TUNING_HPP
#define HOPS_TUNING_HPP

#include <hops/MarkovChain/Tuning/AcceptanceRateTuner.hpp>
#include <hops/MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.hpp>
#include <hops/MarkovChain/Tuning/ThompsonSamplingTuner.hpp>
#include <hops/MarkovChain/Tuning/SimpleExpectedSquaredJumpDistanceTuner.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/Run.hpp>

#include <Eigen/Core>

#include <memory>
#include <random>
#include <stdexcept>
#include <chrono>
#include <unordered_map>


namespace hops {

    struct TuningData {
        std::string method;
        std::string target;
        unsigned long totalNumberOfSamples;
        unsigned long totalNumberOfIterations;
        double tunedObjectiveValue;
        double totalTimeTaken;

        Eigen::MatrixXd data;
        Eigen::MatrixXd posterior;
    };

    template<typename Model, typename Proposal, typename TuningTarget>
    std::tuple<Eigen::VectorXd, TuningData> tune(const RunBase<Model, Proposal>& run, ThompsonSamplingTuner::param_type& parameters, TuningTarget& target) {
        RunBase<Model, Proposal> tuningRun(run);
        Eigen::VectorXd tunedParameters;
        TuningData tuningData;
        
        tuningRun.init();

        // record tuning time 
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        ThompsonSamplingTuner::tune(tunedParameters, 
                                    tuningData.tunedObjectiveValue, 
                                    tuningRun.markovChains, 
                                    tuningRun.randomNumberGenerators, 
                                    parameters,
                                    target,
                                    tuningData.data, 
                                    tuningData.posterior);

        time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count() - time;

        tuningData.method = "ThompsonSampling";
        tuningData.target = target.getName();
        tuningData.totalTimeTaken = time;
        tuningData.totalNumberOfSamples = run.markovChains.size() * parameters.iterationsToTestStepSize * parameters.posteriorUpdateIterationsNeeded * parameters.pureSamplingIterations; 
        tuningData.totalNumberOfIterations = parameters.posteriorUpdateIterationsNeeded * parameters.pureSamplingIterations;

        return {tunedParameters, tuningData};
    }

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, AcceptanceRateTuner::param_type& parameters) {
        if (!run.isInitialized) {
            run.init();
        }

        double tunedStepSize, tunedObjectiveValue;
        Eigen::MatrixXd data, posterior;
        
        // record tuning time 
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        AcceptanceRateTuner::tune(tunedStepSize, 
                                  tunedObjectiveValue, 
                                  run.markovChains, 
                                  run.randomNumberGenerators, 
                                  parameters,
                                  data, 
                                  posterior);

        time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count() - time;


        for (size_t i = 0; i < run.markovChains.size(); ++i) {
            try {
                run.markovChains[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, tunedStepSize);
            } catch (...) {
                //
            }
        }

        run.stepSize = tunedStepSize;
        unsigned long totalNumberOfTuningSamples = 
                run.markovChains.size() * parameters.iterationsToTestStepSize * parameters.posteriorUpdateIterationsNeeded * parameters.pureSamplingIterations; 
        // reset stored states
        run.data->reset();

        run.data->tuningMethod = "ThompsonSamplingAcceptanceRate";
        run.data->totalNumberOfTuningSamples = totalNumberOfTuningSamples;
        run.data->tunedStepSize = tunedStepSize; 
        run.data->tunedObjectiveValue = tunedObjectiveValue;
        run.data->totalTuningTimeTaken = time; 

        run.data->tuningData = data; 
        run.data->tuningPosterior = posterior; 
    }

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
        if (!run.isInitialized) {
            run.init();
        }

        double tunedStepSize, maximumExpectedSquaredJumpDistance;
        Eigen::MatrixXd data, posterior;
        
        // record tuning time 
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        ExpectedSquaredJumpDistanceTuner::tune(tunedStepSize, 
                                               maximumExpectedSquaredJumpDistance, 
                                               run.markovChains, 
                                               run.randomNumberGenerators, 
                                               parameters, 
                                               data, 
                                               posterior);

        time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count() - time;


        for (size_t i = 0; i < run.markovChains.size(); ++i) {
            try {
                run.markovChains[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, tunedStepSize);
            } catch (...) {
                //
            }
        }

        run.stepSize = tunedStepSize;
        unsigned long totalNumberOfTuningSamples = 
                run.markovChains.size() * parameters.iterationsToTestStepSize * parameters.posteriorUpdateIterationsNeeded * parameters.pureSamplingIterations; 
        // reset stored states
        run.data->reset();

        run.data->tuningMethod = "ThompsonSamplingExpectedSquaredJumpDistance";
        run.data->totalNumberOfTuningSamples = totalNumberOfTuningSamples;
        run.data->tunedStepSize = tunedStepSize; 
        run.data->tunedObjectiveValue = maximumExpectedSquaredJumpDistance;
        run.data->totalTuningTimeTaken = time; 

        run.data->tuningData = data; 
        run.data->tuningPosterior = posterior; 
    }

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters) {
        if (!run.isInitialized) {
            run.init();
        }

        double tunedStepSize, maximumExpectedSquaredJumpDistance;
        Eigen::MatrixXd data, posterior;
        
        // record tuning time 
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        SimpleExpectedSquaredJumpDistanceTuner::tune(tunedStepSize, 
                                                      maximumExpectedSquaredJumpDistance, 
                                                      run.markovChains, 
                                                      run.randomNumberGenerators, 
                                                      parameters);

        time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count() - time;


        for (size_t i = 0; i < run.markovChains.size(); ++i) {
            try {
                run.markovChains[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, tunedStepSize);
            } catch (...) {
                //
            }
        }

        run.stepSize = tunedStepSize;
        unsigned long totalNumberOfTuningSamples = 
                run.markovChains.size() * parameters.iterationsToTestStepSize * parameters.stepSizeGridSize; 
        // reset stored states
        run.data->reset();

        run.data->tuningMethod = "GridSearchESJD";
        run.data->totalNumberOfTuningSamples = totalNumberOfTuningSamples;
        run.data->tunedStepSize = tunedStepSize; 
        run.data->tunedObjectiveValue = maximumExpectedSquaredJumpDistance;
        run.data->totalTuningTimeTaken = time; 

        run.data->tuningData = data; 
        run.data->tuningPosterior = posterior; 
    }
}

#endif // HOPS_TUNING_HPP
