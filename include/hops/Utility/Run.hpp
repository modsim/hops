#ifndef HOPS_RUN_HPP
#define HOPS_RUN_HPP

#include "../LinearProgram/LinearProgramClpImpl.hpp"
#include "../LinearProgram/LinearProgramGurobiImpl.hpp"
#include "../MarkovChain/AcceptanceRateTuner.hpp"
#include "../MarkovChain/ExpectedSquaredJumpDistanceTuner.hpp"
#include "../MarkovChain/MarkovChain.hpp"
#include "../MarkovChain/MarkovChainFactory.hpp"
#include "../MarkovChain/MarkovChainType.hpp"
#include "../MarkovChain/SimpleExpectedSquaredJumpDistanceTuner.hpp"
#include "../Polytope/MaximumVolumeEllipsoid.hpp"
#include "../RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "Data.hpp"
#include "Exceptions.hpp"
#include "Problem.hpp"

#include <Eigen/Core>

#include <memory>
#include <random>
#include <stdexcept>

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
    void tune(RunBase<Model, Proposal>& run, const AcceptanceRateTuner::param_type& parameters);

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, const ExpectedSquaredJumpDistanceTuner::param_type& parameters);

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, const SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters);


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

		void setDiagnosticsThreshold(double diagnosticsThreshold);
		double getDiagnosticsThreshold();

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
            	randomNumberGenerators.clear();
            	RandomNumberGenerator rng(randomSeed);
            	std::uniform_int_distribution<unsigned> uniform(std::numeric_limits<unsigned>::min(), std::numeric_limits<unsigned>::max());
            	for (unsigned long i = 0; i < numberOfChains; ++i) {
            		randomNumberGenerators.push_back(RandomNumberGenerator(uniform(rng)));
            	}
            }

            // initialize missing starting points with the chebyshev center or the starting point passed
            // by the problem.
            if (startingPoints.size() < numberOfChains) {
                Eigen::VectorXd chebyshev;
                if (!problem.useStartingPoint) {
#if defined(HOPS_GUROBI_FOUND) || defined(HOPS_CLP_FOUND)
                    try {
                        LinearProgramGurobiImpl linearProgram(problem.A, problem.b);
                        chebyshev = linearProgram.calculateChebyshevCenter().optimalParameters;

                    // either std::runtime_error, if Gurobi wasn't found or GRBException if no license
                    } catch (...) { 
                        LinearProgramClpImpl linearProgram(problem.A, problem.b);
                        chebyshev = linearProgram.calculateChebyshevCenter().optimalParameters;
                    }
#else
                    throw MissingStartingPointsException(
                            "No default starting point was provided in problem and no "
                            "LP solver is available for computing the Chebyshev center, "
                            "can thus not intialize missing starting points.");
#endif
                }

                // initialize all missing starting points
                for (unsigned long i = startingPoints.size(); i < numberOfChains; ++i) {
                    if (problem.useStartingPoint) {
                        startingPoints.push_back(problem.startingPoint);
                    } else {
                        startingPoints.push_back(chebyshev);
                    }
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
                                                                                          problem.model,
                                                                                          false));
                    } else {
                        proposal.setState(startingPoints[i]);
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(proposal,
                                                                                          problem.unroundingTransformation,
                                                                                          problem.unroundingShift,
                                                                                          problem.model,
                                                                                          false));
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
                                                                                          problem.model,
                                                                                          false));
					} else {
                        proposal.setState(roundedStartingPoint);
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(proposal,
                                                                                          roundingTransformation,
                                                                                          Eigen::VectorXd(Eigen::VectorXd::Zero(problem.dimension)),
                                                                                          problem.model,
                                                                                          false));
                    }
				} else {
					if constexpr(std::is_same<Proposal, NoProposal>::value) {
                        markovChains[i] = std::move(MarkovChainFactory::createMarkovChain(markovChainType,
                                                                                          problem.A,
                                                                                          problem.b,
                                                                                          startingPoints[i],
                                                                                          problem.model,
                                                                                          false));
					} else {
                        proposal.setState(startingPoints[i]);
                        markovChains[i] = std::move(
                                MarkovChainFactory::createMarkovChain<Eigen::MatrixXd, Eigen::VectorXd, Model, Proposal>(proposal,
                                                                                                                         problem.model,
                                                                                                                         false));
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
            double convergenceDiagnostics = 0;
            do {
                #pragma omp parallel for
                for (unsigned long i = 0; i < numberOfChains; ++i) {
                    markovChains[i]->draw(randomNumberGenerators[i], numberOfSamples, thinning);
                }
                // end pragma omp parallel for

                if (sampleUntilConvergence && numberOfChains >= 2) {
                    convergenceDiagnostics = computePotentialScaleReductionFactor(*data).maxCoeff();
                }

                //numSeen += conf.numSamples;
                ++k;
            } while(sampleUntilConvergence &&
                    // if threshold was not met or if psrf is nan, keep going
                    (convergenceDiagnostics > diagnosticsThreshold || std::isnan(convergenceDiagnostics))  &&
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
        double diagnosticsThreshold = 1.05;
        unsigned long maxRepetitions = 1;

        double stepSize = 1;
        double fisherWeight = 0.5;

        double randomSeed = 0;

        //friend void tune(RunBase<Model, Proposal>& run, const ExpectedSquaredJumpDistanceTuner::param_type& parameters);
        friend void tune<>(RunBase& run, const AcceptanceRateTuner::param_type& parameters);
        friend void tune<>(RunBase& run, const ExpectedSquaredJumpDistanceTuner::param_type& parameters);
        friend void tune<>(RunBase& run, const SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters);
    };

    template <typename Model>
    using Run = RunBase<Model, NoProposal>;

	template<typename Model, typename Proposal>
	void RunBase<Model, Proposal>::setProblem(const Problem<Model>& problem) {
		this->isInitialized = false;
		this->problem = &problem;
	}

	template<typename Model, typename Proposal>
	const Problem<Model>& RunBase<Model, Proposal>::getProblem() {
		return *problem;
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
    void RunBase<Model, Proposal>::setDiagnosticsThreshold(double diagnosticsThreshold) {
		this->diagnosticsThreshold = diagnosticsThreshold;
	}

	template<typename Model, typename Proposal>
	double RunBase<Model, Proposal>::getDiagnosticsThreshold() {
		return diagnosticsThreshold;
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
    void tune(RunBase<Model, Proposal>& run, const AcceptanceRateTuner::param_type& parameters) {
        if (!run.isInitialized) {
            run.init();
        }

        double tunedStepSize, acceptanceRate;
        
        // record tuning time 
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        AcceptanceRateTuner::tune(tunedStepSize, 
                                  acceptanceRate, 
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
                parameters.maximumTotalIterations; 
        // reset stored states
        run.data->reset();
        run.data->setTuningData(totalNumberOfTuningSamples, tunedStepSize, -1, acceptanceRate, time);
    }

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, const ExpectedSquaredJumpDistanceTuner::param_type& parameters) {
        if (!run.isInitialized) {
            run.init();
        }

        double tunedStepSize, maximumExpectedSquaredJumpDistance;
        
        // record tuning time 
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();

        ExpectedSquaredJumpDistanceTuner::tune(tunedStepSize, 
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
                run.markovChains.size() * parameters.iterationsToTestStepSize * parameters.maximumTotalIterations; 
        // reset stored states
        run.data->reset();
        run.data->setTuningData(totalNumberOfTuningSamples, tunedStepSize, maximumExpectedSquaredJumpDistance, -1, time);
    }

    template<typename Model, typename Proposal>
    void tune(RunBase<Model, Proposal>& run, const SimpleExpectedSquaredJumpDistanceTuner::param_type& parameters) {
        if (!run.isInitialized) {
            run.init();
        }

        double tunedStepSize, maximumExpectedSquaredJumpDistance;
        
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
                run.markovChains.size() * parameters.iterationsToTestStepSize; 
        // reset stored states
        run.data->reset();
        run.data->setTuningData(totalNumberOfTuningSamples, tunedStepSize, maximumExpectedSquaredJumpDistance, -1, time);
    }
}

#endif // HOPS_RUN_HPP
