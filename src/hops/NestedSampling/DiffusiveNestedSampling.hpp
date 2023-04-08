#ifndef HOPS_DIFFUSIVENESTEDSAMPLING_HPP
#define HOPS_DIFFUSIVENESTEDSAMPLING_HPP

#include <thread>
#include <vector>

#include "hops/MarkovChain/Proposal/Proposal.hpp"
#include "hops/Model/Model.hpp"

#include "LogLikelihoodValue.hpp"
#include "Level.hpp"

namespace hops {


    struct DiffusiveNestedSamplingOptions {
        double compression;
        // Numerical options
        unsigned int new_level_interval;
//        unsigned int save_interval; this is output thinning
        unsigned int thread_steps;
        unsigned int max_num_levels;
        double lambda;
        double beta;
//        unsigned int max_num_saves;


        bool adaptive;
    };

    /**
     * @brief Algorithm from https://arxiv.org/pdf/1606.03757.pdf
     */
    class DiffusiveNestedSampling {
    public:
        struct State {
            VectorType modelparameters;
            double levelAssigment;
            double likelihood;
            double tiebreaker;
            size_t particleId;

            // output after each step: a random particle with:
            // params +level assignment, log likelihood, tiebreaker, ID.
        };


        DiffusiveNestedSampling(std::vector<std::unique_ptr<Proposal>> particles,
                                std::vector<std::unique_ptr<Model>> models,
                                DiffusiveNestedSamplingOptions options) {
        }

        std::vector<std::string> getDimensionNames(); // param names + level assigment + log like + tiebreaker + ID

        VectorType saveRandomParticle(RandomNumberGenerator &rng);

        VectorType saveParticle(size_t particleIndex);

        std::vector<Level> getLevels();

        void updateParticles(RandomNumberGenerator &rng);

        void updateParticle(size_t particleIndex, RandomNumberGenerator &rngs);

        void updateLevelAssigment(size_t particleIndex);

        void recalculateLogX();

        void renormalizeVisits();

        void updateLevelAssigment(int which);

        void killLaggingParticles();

        bool hasFinishedLevelConstruction();

        bool isLevelSpacingSmallEnough();

        double log_push(unsigned int which_level) const; // weighting function


//        void Level::recalculate_log_X(vector<Level>& levels, double compression,
//                                      unsigned int regularisation)
//        {
//            assert(levels.size() > 0);
//
//            levels[0].log_X = 0.0;
//            for(size_t i=1; i<levels.size(); ++i)
//                levels[i].log_X = levels[i-1].log_X + log((double)(levels[i-1].exceeds + (1./compression)*regularisation)/(double)(levels[i-1].visits + regularisation));
//        }
//
//        void Level::renormalise_visits(vector<Level>& levels,
//                                       unsigned int regularisation)
//        {
//            for(auto& level: levels)
//            {
//                if(level.tries >= regularisation)
//                {
//                    level.accepts = ((double)(level.accepts+1)/(double)(level.tries+1))*regularisation;
//                    level.tries = regularisation;
//                }
//                if(level.visits >= regularisation)
//                {
//                    level.exceeds = ((double)(level.exceeds+1)/(double)(level.visits+1))*regularisation;
//                    level.visits = regularisation;
//                }
//            }
//        }
//

    private:
        std::vector<std::unique_ptr<Proposal>> particles;
        std::vector<std::unique_ptr<Model>> models;
        DiffusiveNestedSamplingOptions options;

        double count_saves;
        double count_mcmc_steps_since_save;
        double count_mcmc_steps;


        // used for adaptive part
        double difficulty;
        double work_ratio;

        std::vector<LogLikelihoodValue> all_above; // new_level_storage
        std::vector<std::vector<LogLikelihoodValue>> above; // storage for likelihoods above threshold

        std::vector<Level> levels;
        std::vector<std::vector<Level>> level_copies;


    };
}


#endif //HOPS_DIFFUSIVENESTEDSAMPLING_HPP
