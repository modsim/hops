#ifndef HOPS_RUN_HPP
#define HOPS_RUN_HPP

#include <hops/LinearProgram/LinearProgramClpImpl.hpp>
#include <hops/LinearProgram/LinearProgramGurobiImpl.hpp>
#include <hops/MarkovChain/MarkovChain.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/MarkovChainType.hpp>
#include <hops/MarkovChain/Tuning/AcceptanceRateTuner.hpp>
#include <hops/MarkovChain/Tuning/ExpectedSquaredJumpDistanceTuner.hpp>
#include <hops/MarkovChain/Tuning/SimpleExpectedSquaredJumpDistanceTuner.hpp>
#include <hops/Polytope/MaximumVolumeEllipsoid.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Utility/Data.hpp>
#include <hops/Utility/Exceptions.hpp>
#include <hops/Utility/Problem.hpp>

#include <Eigen/Core>

#include <memory>
#include <random>
#include <stdexcept>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif 

namespace hops {
    struct NoProposal {
        void setState(Eigen::VectorXd) {
            throw std::runtime_error("NoProposal.setState was called, but this class is not actually supposed to be used.");
        }
    };

    template<typename Model, typename Proposal>
    class RunBase;


    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, AcceptanceRateTuner::param_type& parameters);

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, ExpectedSquaredJumpDistanceTuner::param_type& parameters);

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters);


    template<typename Model, typename Proposal>
    class RunBase {
    public:
        RunBase (MarkovChainType markovChainType,
                 unsigned long numberOfSamples = 1000, 
                 unsigned long numberOfChains = 1) :
                markovChainType(markovChainType),
                numberOfChains(numberOfChains), 
                numberOfSamples(numberOfSamples) {
            //
        }

        RunBase (const Problem<Model>& problem, 
                 MarkovChainType markovChainType = MarkovChainType::HitAndRun,
                 unsigned long numberOfSamples = 1000, 
                 unsigned long numberOfChains = 1) :
                problem(problem),
                markovChainType(markovChainType),
                numberOfChains(numberOfChains), 
                numberOfSamples(numberOfSamples) {
            //
        }
	
        RunBase (Proposal proposal,
                 unsigned long numberOfSamples = 1000, 
                 unsigned long numberOfChains = 1) :
                proposal(proposal),
                numberOfChains(numberOfChains), 
                numberOfSamples(numberOfSamples) {
            //
        }

        RunBase (const Problem<Model>& problem, 
                 Proposal proposal,
                 unsigned long numberOfSamples = 1000, 
                 unsigned long numberOfChains = 1) :
                problem(problem),
                proposal(proposal),
                numberOfChains(numberOfChains), 
                numberOfSamples(numberOfSamples) {
            //
        }
	
		RunBase() = default;

		void setProblem(const Problem<Model>& problem);
		const Problem<Model>& getProblem();

		void setStartingPoints(const std::vector<Eigen::VectorXd>& startingPoints);
		const std::vector<Eigen::VectorXd>& getStartingPoints();

		void setMarkovChainType(MarkovChainType markovChainType);
		MarkovChainType getMarkovChainType();

		void setNumberOfChains(unsigned long numberOfChains);
		unsigned long getNumberOfChains();

		void setNumberOfSamples(unsigned long numberOfSamples);
		unsigned long getNumberOfSamples();

		void setThinning(unsigned long thinning);
		unsigned long getThinning();

        void setUseRounding(bool useRounding);
        bool getUseRounding();

		void setStepSize(double stepSize);
		double getStepSize();

		void setFisherWeight(double fisherWeight);
		double getFisherWeight();

		void setRandomSeed(double randomSeed);
		double getRandomSeed();

		void setSamplingUntilConvergence(bool sampleUntilConvergence);
		bool getSamplingUntilConvergence();

		void setConvergenceThreshold(double convergenceThreshold);
		double getConvergenceThreshold();

		void setMaxRepetitions(double maxRepetitions);
		double getMaxRepetitions();

		Data& getData();

        /**
         *
         */
        void init() {
            // create new data object to not mix up possible previous data
            data = std::make_shared<Data>(problem.dimension);

            if (!isRandomGeneratorInitialized) {
            	isRandomGeneratorInitialized = true;
            	// initialize random number generator for each chain
            	for (unsigned long i = 0; i < numberOfChains; ++i) {
            		randomNumberGenerators.push_back(RandomNumberGenerator(randomSeed, i));
            	}
            }

            // initialize missing starting points with the chebyshev center or the starting point passed
            // by the problem.
            if (startingPoints.size() < numberOfChains) {
                Eigen::VectorXd defaultStartingPoint;
                if (!problem.useStartingPoint && startingPoints.size() == 0) {
#if defined(HOPS_GUROBI_FOUND) || defined(HOPS_CLP_FOUND)
                    try {
                        LinearProgramGurobiImpl linearProgram(problem.A, problem.b);
                        defaultStartingPoint = linearProgram.computeChebyshevCenter().optimalParameters;

                    // either std::runtime_error, if Gurobi wasn't found or GRBException if no license
                    } catch (...) { 
                        LinearProgramClpImpl linearProgram(problem.A, problem.b);
                        defaultStartingPoint = linearProgram.computeChebyshevCenter().optimalParameters;
                    }
#else
                    throw MissingStartingPointsException(
                            "No default starting point was provided in problem and no "
                            "LP solver is available for computing the Chebyshev center, "
                            "can thus not intialize missing starting points.");
#endif
                } if (problem.useStartingPoint) {
                    defaultStartingPoint = problem.startingPoint;
                } else {
                    defaultStartingPoint = startingPoints.back();
                }

                // initialize all missing starting points
                for (unsigned long i = startingPoints.size(); i < numberOfChains; ++i) {
                    startingPoints.push_back(defaultStartingPoint);
                }
            }

            // if rounding was specified, then compute the round transformation
			Eigen::MatrixXd roundingTransformation;
			if (useRounding) {
				roundingTransformation =
					hops::MaximumVolumeEllipsoid<double>::construct(problem.A, problem.b, 10000)
					.getRoundingTransformation();
				// next transform of startingPoint assumes roundingTransformation is lower triangular
				if (!roundingTransformation.isLowerTriangular()) {
					throw std::runtime_error("Error while rounding starting point, check code.");
				}
			}

            // set up the chains with the problem specifications
            markovChains.resize(numberOfChains);
            for (unsigned long i = 0; i < numberOfChains; ++i) {
                if (problem.unround) {
					if constexpr(std::is_same<Proposal, NoProposal>::value) {
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(markovChainType,
                                                                                          problem.A,
                                                                                          problem.b,
                                                                                          startingPoints[i],
                                                                                          problem.unroundingTransformation,
                                                                                          problem.unroundingShift,
                                                                                          problem.model));
                    } else {
                        proposal.setState(startingPoints[i]);
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(proposal,
                                                                                          problem.unroundingTransformation,
                                                                                          problem.unroundingShift,
                                                                                          problem.model));
                    }
                } else if (useRounding) {
                    Eigen::VectorXd roundedStartingPoint = roundingTransformation.triangularView<Eigen::Lower>().solve(startingPoints[i]);
					if constexpr(std::is_same<Proposal, NoProposal>::value) {
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(markovChainType,
                                                                                          Eigen::MatrixXd(problem.A * roundingTransformation),
                                                                                          problem.b,
                                                                                          roundedStartingPoint,
                                                                                          roundingTransformation,
                                                                                          Eigen::VectorXd(Eigen::VectorXd::Zero(problem.dimension)),
                                                                                          problem.model));
					} else {
                        proposal.setState(roundedStartingPoint);
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(proposal,
                                                                                          roundingTransformation,
                                                                                          Eigen::VectorXd(Eigen::VectorXd::Zero(problem.dimension)),
                                                                                          problem.model));
                    }
				} else {
					if constexpr(std::is_same<Proposal, NoProposal>::value) {
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(markovChainType,
                                                                                          problem.A,
                                                                                          problem.b,
                                                                                          startingPoints[i],
                                                                                          problem.model));
					} else {
                        proposal.setState(startingPoints[i]);
                        markovChains[i] = std::move(
                                MarkovChainFactory::createMarkovChain<Eigen::MatrixXd, Eigen::VectorXd, Model, Proposal>(proposal,
                                                                                                                         problem.model));
                    }
                }

                // preallocate records vector to avoid reallocation
                markovChains[i]->clearHistory();
                markovChains[i]->reserveStateRecords(maxRepetitions * numberOfSamples); 

                try {
                    markovChains[i]->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, stepSize);
                } catch (...) {
                    //
                }

                try {
                    markovChains[i]->setAttribute(hops::MarkovChainAttribute::FISHER_WEIGHT, fisherWeight);
                } catch (...) {
                    //
                }
            }

            // register data object with chains. this sets the pointers in the data
            // object to the chain's data vectors, making the data outlive the chain.
            data->linkWithChains(markovChains);
            data->setDimension(problem.dimension);

            isInitialized = true;
        }

        /**
         *
         */
        void sample() {
			sample(numberOfSamples, thinning);	
		}	

        /**
         *
         */
        void sample(unsigned long numberOfSamples, unsigned long thinning = 1) {
            if (!isInitialized) {
                init();
            }

            unsigned long k = 0;
            double convergenceStatistics = 0;
            do {
                #pragma omp parallel for
                for (unsigned long i = 0; i < numberOfChains; ++i) {
                    markovChains[i]->draw(randomNumberGenerators[i], numberOfSamples, thinning);
                }
                // end pragma omp parallel for

                if (sampleUntilConvergence && numberOfChains >= 2) {
                    convergenceStatistics = computePotentialScaleReductionFactor(*data).maxCoeff();
                }

                //numSeen += conf.numSamples;
                ++k;
            } while(sampleUntilConvergence &&
                    // if threshold was not met or if psrf is nan, keep going
                    (convergenceStatistics > convergenceThreshold || std::isnan(convergenceStatistics))  &&
                    // though only if we have not yet reached the maximum number of repetitions
                    k < maxRepetitions);
        }

    private:
        Problem<Model> problem;
        std::shared_ptr<Data> data = nullptr;

        Proposal proposal;

        bool isInitialized = false;
        bool isRandomGeneratorInitialized = false;

        MarkovChainType markovChainType = MarkovChainType::HitAndRun;

        std::vector<std::shared_ptr<hops::MarkovChain>> markovChains;
        std::vector<hops::RandomNumberGenerator> randomNumberGenerators;

        std::vector<Eigen::VectorXd> startingPoints;

        unsigned long numberOfChains = 1;
        unsigned long numberOfSamples = 1000;
        unsigned long thinning = 1;

        bool useRounding = false;

        bool sampleUntilConvergence = false;
        double convergenceThreshold = 1.05;
        unsigned long maxRepetitions = 1;

        double stepSize = 1;
        double fisherWeight = 0.5;

        double randomSeed = 0;

        //friend void tune(RunBase<Model, Proposal>& run, const ExpectedSquaredJumpDistanceTuner::param_type& parameters);
        friend void tune<>(RunBase& run, AcceptanceRateTuner::param_type& parameters);
        friend void tune<>(RunBase& run, ExpectedSquaredJumpDistanceTuner::param_type& parameters);
        friend void tune<>(RunBase& run, SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters);
    };

    template <typename Model>
    using Run = RunBase<Model, NoProposal>;

	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setProblem(const Problem<Model>& problem) {
		this->isInitialized = false;
		this->problem = problem;
	}

	template<typename Model, typename Proposal>
	const Problem<Model>& RunBase<Model, Proposal>::getProblem() {
		return problem;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setStartingPoints(const std::vector<Eigen::VectorXd>& startingPoints) {
		this->isInitialized = false;
		this->startingPoints = startingPoints;
	}

	template<typename Model, typename Proposal>
	const std::vector<Eigen::VectorXd>& RunBase<Model, Proposal>::getStartingPoints() {
		return startingPoints;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setMarkovChainType(MarkovChainType markovChainType) {
		this->isInitialized = false;
		this->markovChainType = markovChainType;
	}

	template<typename Model, typename Proposal>
	MarkovChainType RunBase<Model, Proposal>::getMarkovChainType() {
		return markovChainType;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setNumberOfChains(unsigned long numberOfChains) {
		this->isInitialized = false;
		this->numberOfChains = numberOfChains;
	}

	template<typename Model, typename Proposal>
	unsigned long RunBase<Model, Proposal>::getNumberOfChains() {
		return numberOfChains;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setNumberOfSamples(unsigned long numberOfSamples) {
		this->numberOfSamples = numberOfSamples;
	}

	template<typename Model, typename Proposal>
	unsigned long RunBase<Model, Proposal>::getNumberOfSamples() {
		return numberOfSamples;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setThinning(unsigned long thinning) {
		this->thinning = thinning;
	}

	template<typename Model, typename Proposal>
	unsigned long RunBase<Model, Proposal>::getThinning() {
		return thinning;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setUseRounding(bool useRounding) {
		this->isInitialized = false;
        this->useRounding = useRounding;
	}

	template<typename Model, typename Proposal>
	bool RunBase<Model, Proposal>::getUseRounding() {
        return this->useRounding;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setStepSize(double stepSize) {
		this->isInitialized = false;
		this->stepSize = stepSize;
	}

	template<typename Model, typename Proposal>
	double RunBase<Model, Proposal>::getStepSize() {
		return stepSize;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setFisherWeight(double fisherWeight) {
		this->isInitialized = false;
		this->fisherWeight = fisherWeight;
	}

	template<typename Model, typename Proposal>
	double RunBase<Model, Proposal>::getFisherWeight() {
		return fisherWeight;
	}


	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setRandomSeed(double randomSeed) {
		this->isInitialized = false;
		this->isRandomGeneratorInitialized = false;
		this->randomSeed = randomSeed;
	}

	template<typename Model, typename Proposal>
	double RunBase<Model, Proposal>::getRandomSeed() {
		return randomSeed;
	}


	template<typename Model, typename Proposal>
    void RunBase<Model, Proposal>::setSamplingUntilConvergence(bool sampleUntilConvergence) {
		this->sampleUntilConvergence = sampleUntilConvergence;
    }

	template<typename Model, typename Proposal>
    bool RunBase<Model, Proposal>::getSamplingUntilConvergence() {
        return this->sampleUntilConvergence;
    }


	template<typename Model, typename Proposal>
    void RunBase<Model, Proposal>::setConvergenceThreshold(double convergenceThreshold) {
		this->convergenceThreshold = convergenceThreshold;
	}

	template<typename Model, typename Proposal>
	double RunBase<Model, Proposal>::getConvergenceThreshold() {
		return convergenceThreshold;
	}


	template<typename Model, typename Proposal>
    void RunBase<Model, Proposal>::setMaxRepetitions(double maxRepetitions) {
        this->maxRepetitions = maxRepetitions;
        if (this->isInitialized) {
            for (unsigned long i = 0; i < numberOfChains; ++i) {
                markovChains[i]->reserveStateRecords(maxRepetitions * numberOfSamples); 
            }
        }
	}

	template<typename Model, typename Proposal>
	double RunBase<Model, Proposal>::getMaxRepetitions() {
		return maxRepetitions;
	}


	template<typename Model, typename Proposal>
	Data& RunBase<Model, Proposal>::getData() {
		return *data;
	}

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, AcceptanceRateTuner::param_type& parameters) {
        if (!run.isInitialized) {
            run.init();
        }

        double tunedStepSize, deltaAcceptanceRate;
        Eigen::MatrixXd data, posterior;
        
        // record tuning time 
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        AcceptanceRateTuner::tune(tunedStepSize, 
                                  deltaAcceptanceRate, 
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

        run.data->setTuningMethod("ThompsonSamplingESJD");
        run.data->setTotalNumberOfTuningSamples(totalNumberOfTuningSamples);
        run.data->setTunedStepSize(tunedStepSize); 
        run.data->setTunedObjectiveValue(deltaAcceptanceRate);
        run.data->setTotalTuningTimeTaken(time); 

        run.data->setTuningData(data); 
        run.data->setTuningPosterior(posterior); 
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

        run.data->setTuningMethod("ThompsonSamplingAcceptanceRate");
        run.data->setTotalNumberOfTuningSamples(totalNumberOfTuningSamples);
        run.data->setTunedStepSize(tunedStepSize); 
        run.data->setTunedObjectiveValue(maximumExpectedSquaredJumpDistance);
        run.data->setTotalTuningTimeTaken(time); 

        run.data->setTuningData(data); 
        run.data->setTuningPosterior(posterior); 
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

        run.data->setTuningMethod("GridSearchESJD");
        run.data->setTotalNumberOfTuningSamples(totalNumberOfTuningSamples);
        run.data->setTunedStepSize(tunedStepSize); 
        run.data->setTunedObjectiveValue(maximumExpectedSquaredJumpDistance);
        run.data->setTotalTuningTimeTaken(time); 

        run.data->setTuningData(data); 
        run.data->setTuningPosterior(posterior); 
    }
}

#endif // HOPS_RUN_HPP
