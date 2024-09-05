#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DNest4AdapterTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>

#include "hops/MarkovChain/ModelMixin.hpp"
#include "hops/MarkovChain/Proposal/ProposalParameter.hpp"
#include "hops/MarkovChain/Proposal/HitAndRunProposal.hpp"
#include "hops/Model/Gaussian.hpp"
#include "hops/NestedSampling/DNest4Adapter.hpp"
#include "hops/MarkovChain/Proposal/ReversibleJumpProposal.hpp"
#include "hops/Model/JumpableModel.hpp"

namespace {

    void check_equal(const DNest4::Sampler<hops::DNest4Adapter> sampler,
                     const DNest4::Sampler<hops::DNest4Adapter> loaded_sampler,
                     int num_threads) {

        auto rngs = sampler.get_rngs();
        auto loaded_rngs = loaded_sampler.get_rngs();
        BOOST_ASSERT(rngs.size() == loaded_rngs.size());
        for(size_t i=0; i<rngs.size(); ++i) {
            BOOST_CHECK(rngs[i].engine.getState() == loaded_rngs[i].engine.getState());
            BOOST_CHECK(rngs[i].engine.getStream() == loaded_rngs[i].engine.getStream());
        }

        auto loaded_likelihoods = loaded_sampler.get_log_likelihoods();
        auto likelihoods = sampler.get_log_likelihoods();
        BOOST_ASSERT(likelihoods.size() == loaded_likelihoods.size());
        for (size_t i = 0; i < loaded_likelihoods.size(); ++i) {
            BOOST_CHECK(loaded_likelihoods[i].get_value() == likelihoods[i].get_value());
            BOOST_CHECK(loaded_likelihoods[i].get_tiebreaker() == likelihoods[i].get_tiebreaker());
        }

        auto loaded_level_assignments = loaded_sampler.get_level_assignments();
        auto level_assignments = sampler.get_level_assignments();
        BOOST_ASSERT(level_assignments.size() == loaded_level_assignments.size());
        for (size_t i = 0; i < loaded_level_assignments.size(); ++i) {
            BOOST_CHECK(loaded_level_assignments[i] == level_assignments[i]);
        }

        auto loaded_levels = loaded_sampler.get_levels();
        auto levels = sampler.get_levels();
        BOOST_ASSERT(levels.size() == loaded_levels.size());
        for (size_t i = 0; i < loaded_levels.size(); ++i) {
            BOOST_CHECK(
                    loaded_levels[i].get_log_likelihood().get_value() == levels[i].get_log_likelihood().get_value());
            BOOST_CHECK(loaded_levels[i].get_log_likelihood().get_tiebreaker() ==
                        levels[i].get_log_likelihood().get_tiebreaker());
            BOOST_CHECK(loaded_levels[i].get_log_X() == levels[i].get_log_X());
            BOOST_CHECK(loaded_levels[i].get_visits() == levels[i].get_visits());
            BOOST_CHECK(loaded_levels[i].get_exceeds() == levels[i].get_exceeds());
            BOOST_CHECK(loaded_levels[i].get_accepts() == levels[i].get_accepts());
            BOOST_CHECK(loaded_levels[i].get_tries() == levels[i].get_tries());
        }

        for (int i = 0; i < num_threads; ++i) {
            BOOST_CHECK(sampler.get_particles()[i].proposal->getState() ==
                        loaded_sampler.get_particles()[i].proposal->getState());
            BOOST_CHECK(sampler.get_particles()[i].proposal->getProposal() ==
                        loaded_sampler.get_particles()[i].proposal->getProposal());
        }

        auto all_above = sampler.all_above;
        auto loaded_all_above = sampler.all_above;
        BOOST_ASSERT(all_above.size() == loaded_all_above.size());
        for(size_t i=0; i<all_above.size(); ++i) {
            BOOST_CHECK(all_above[i].get_value() == loaded_all_above[i].get_value());
            BOOST_CHECK(all_above[i].get_tiebreaker() == loaded_all_above[i].get_tiebreaker());
        }


        auto above = sampler.above;
        auto loaded_above = loaded_sampler.above;
        BOOST_ASSERT(above.size() == loaded_above.size());
        for(size_t i=0; i<above.size(); ++i) {
            BOOST_CHECK(above[i].size() == loaded_above[i].size());
            for(size_t j=0; j<above[i].size(); ++j) {
                BOOST_CHECK(above[i][j].get_value() == loaded_above[i][j].get_value());
                BOOST_CHECK(above[i][j].get_tiebreaker() == loaded_above[i][j].get_tiebreaker());
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE(DNest4AdapterTestSuite)

BOOST_AUTO_TEST_CASE(TestSerialization) {
        std::cout.clear();
        std::cout.setstate(std::ios_base::failbit);
        int num_threads = 1;

        hops::VectorType mean(1);
        mean << 0;
        Eigen::MatrixXd covariance(1, 1);
        covariance << 1;
        // constrain to [-5, 5]
        Eigen::MatrixXd A(2, 1);
        A << 1, -1;
        hops::VectorType b(2);
        b << 5, 5;

        std::vector<hops::VectorType> priorSamples;
        priorSamples.reserve(16);
        for (long i=0; i<16; ++i) {
            priorSamples.emplace_back(mean);
        }

        auto gaussian = std::make_unique<hops::Gaussian>(mean, covariance);

        std::unique_ptr<hops::Proposal> proposer =
        std::make_unique<hops::HitAndRunProposal<decltype(A), decltype(b),
        hops::GaussianStepDistribution<double>, true>>(A, b, mean);
        proposer->setParameter(hops::ProposalParameter::STEP_SIZE, 0.5);

        hops::DNest4EnvironmentSingleton::getInstance().setProposal(std::move(proposer));
        hops::DNest4EnvironmentSingleton::getInstance().setModel(std::move(gaussian));
        hops::DNest4EnvironmentSingleton::getInstance().setPriorSamples(priorSamples);

        DNest4::Options sampler_options(num_threads, // num_particles
        100, // new_level_interval
        100, // save_interval
        200, // thread_steps
        80, // max_num_levels
        10, // lambda
        100, // beta
        50, // max_num_saves
        false // use  hexfloat
        );

        DNest4::Sampler<hops::DNest4Adapter> sampler(num_threads, // threads
        2.7182818284590451, // compression double
        sampler_options,
        true, // save to disk
        true // adaptive
        );

        sampler.initialise(42);
        sampler.run();
        sampler.save_checkpoint();

        DNest4::Sampler<hops::DNest4Adapter> loaded_sampler(num_threads,
        2.718281828459045, // compression double
        sampler_options,
        false, // save to disk
        true // adaptive
        );
        loaded_sampler.read_checkpoint();
        check_equal(sampler, loaded_sampler, num_threads);

        sampler.increase_max_num_saves(1000);
        loaded_sampler.increase_max_num_saves(1000);
        sampler.run();
        loaded_sampler.run();
        check_equal(sampler, loaded_sampler, num_threads);
}

BOOST_AUTO_TEST_CASE(TestSerializationRJMCMC) {
    std::cout.clear();
    std::cout.setstate(std::ios_base::failbit);
    int num_threads = 1;

    hops::VectorType mean(2);
    mean << 0, 1;
    Eigen::MatrixXd covariance(2, 2);
    covariance << 1, 0, 0, 1;
    // constrain to [-5, 5]
    Eigen::MatrixXd A(4, 2);
    A << 1, 0, 0, 1, -1, 0, 0, -1;
    hops::VectorType b(4);
    b << 5, 5, 5, 5;

    std::vector<hops::VectorType> priorSamples;
    hops::VectorType prior_sample(4);
    prior_sample << 1, 0, 0, 1;
    priorSamples.reserve(16);
    for (long i=0; i<16; ++i) {
        priorSamples.emplace_back(prior_sample);
    }

    auto gaussian = std::make_unique<hops::Gaussian>(mean, covariance);
    auto jumpableModel = std::make_unique<hops::JumpableModel<std::unique_ptr<hops::Model>>>(std::move(gaussian));

    std::unique_ptr<hops::Proposal> proposer =
            std::make_unique<hops::HitAndRunProposal<decltype(A), decltype(b),
                    hops::GaussianStepDistribution<double>, true>>(A, b, mean);
    proposer->setParameter(hops::ProposalParameter::STEP_SIZE, 0.5);

    Eigen::VectorXi jumpIndices(1);
    jumpIndices << 1;
    Eigen::VectorXd defaultValues(1);
    defaultValues << 0;

    auto rjmcmc_proposal = std::make_unique<hops::ReversibleJumpProposal>(proposer->copyProposal(),
                                 jumpIndices,
                                 defaultValues,
                                 A,
                                 b);

    hops::DNest4EnvironmentSingleton::getInstance().setProposal(std::move(rjmcmc_proposal));
    hops::DNest4EnvironmentSingleton::getInstance().setModel(std::move(jumpableModel));
    hops::DNest4EnvironmentSingleton::getInstance().setPriorSamples(priorSamples);

    DNest4::Options sampler_options(num_threads, // num_particles
                                    100, // new_level_interval
                                    50, // save_interval
                                    10, // thread_steps
                                    80, // max_num_levels
                                    10, // lambda
                                    100, // beta
                                    50, // max_num_saves
                                    false // use  hexfloat
    );

    DNest4::Sampler<hops::DNest4Adapter> sampler(num_threads, // threads
                                                 2.7182818284590451, // compression double
                                                 sampler_options,
                                                 false, // save to disk
                                                 true // adaptive
    );

    sampler.initialise(42);
    sampler.run();
    sampler.save_checkpoint();

    DNest4::Sampler<hops::DNest4Adapter> loaded_sampler(num_threads,
                                                        2.718281828459045, // compression double
                                                        sampler_options,
                                                        false, // save to disk
                                                        true // adaptive
    );
    loaded_sampler.read_checkpoint();
    check_equal(sampler, loaded_sampler, num_threads);

    sampler.increase_max_num_saves(1000);
    loaded_sampler.increase_max_num_saves(1000);
    sampler.run();
    std::cout << std::endl;
    loaded_sampler.run();
    check_equal(sampler, loaded_sampler, num_threads);
}

BOOST_AUTO_TEST_CASE(TestSerializationRJMCMC2) {
        std::cout.clear();
        std::cout.setstate(std::ios_base::failbit);
        int num_threads = 1;

        hops::VectorType mean(2);
        mean << 0, 1;
        Eigen::MatrixXd covariance(2, 2);
        covariance << 1, 0, 0, 1;
        // constrain to [-5, 5]
        Eigen::MatrixXd A(4, 2);
        A << 1, 0, 0, 1, -1, 0, 0, -1;
        hops::VectorType b(4);
        b << 5, 5, 5, 5;

        std::vector<hops::VectorType> priorSamples;
        hops::VectorType prior_sample(4);
        prior_sample << 1, 0, 0, 1;
        priorSamples.reserve(16);
        for (long i=0; i<16; ++i) {
            priorSamples.emplace_back(prior_sample);
        }

        auto gaussian = std::make_unique<hops::Gaussian>(mean, covariance);
        auto jumpableModel = std::make_unique<hops::JumpableModel<std::unique_ptr<hops::Model>>>(std::move(gaussian));

        std::unique_ptr<hops::Proposal> proposer =
                std::make_unique<hops::HitAndRunProposal<decltype(A), decltype(b),
                        hops::GaussianStepDistribution<double>, true>>(A, b, mean);
        proposer->setParameter(hops::ProposalParameter::STEP_SIZE, 0.5);

        Eigen::VectorXi jumpIndices(1);
        jumpIndices << 1;
        Eigen::VectorXd defaultValues(1);
        defaultValues << 0;

        auto rjmcmc_proposal = std::make_unique<hops::ReversibleJumpProposal>(proposer->copyProposal(),
                                                                              jumpIndices,
                                                                              defaultValues,
                                                                              A,
                                                                              b);

        hops::DNest4EnvironmentSingleton::getInstance().setProposal(std::move(rjmcmc_proposal));
        hops::DNest4EnvironmentSingleton::getInstance().setModel(std::move(jumpableModel));
        hops::DNest4EnvironmentSingleton::getInstance().setPriorSamples(priorSamples);

        DNest4::Options sampler_options1(num_threads, // num_particles
                                        100, // new_level_interval
                                        50, // save_interval
                                        10, // thread_steps
                                        80, // max_num_levels
                                        10, // lambda
                                        100, // beta
                                        500, // max_num_saves
                                        false // use  hexfloat
        );

        DNest4::Sampler<hops::DNest4Adapter> sampler1(num_threads, // threads
                                                     2.718281828459045, // compression double
                                                     sampler_options1,
                                                     false, // save to disk
                                                     true // adaptive
        );

        sampler1.initialise(42);
        sampler1.run();

        DNest4::Options sampler_options2(num_threads, // num_particles
                                          100, // new_level_interval
                                          50, // save_interval
                                          10, // thread_steps
                                          80, // max_num_levels
                                          10, // lambda
                                          100, // beta
                                          10, // max_num_saves
                                          false // use  hexfloat
        );

        DNest4::Sampler<hops::DNest4Adapter> sampler2(num_threads,
                                                            2.718281828459045, // compression double
                                                            sampler_options2,
                                                            false, // save to disk
                                                            true // adaptive
        );

        sampler2.initialise(42);
        sampler2.run();


        DNest4::Sampler<hops::DNest4Adapter> loaded_sampler2(num_threads,
                                                             2.718281828459045, // compression double
                                                             // loads options1, because it has increased number of saves
                                                             sampler_options1,
                                                             false, // save to disk
                                                             true // adaptive
                                                             );
        loaded_sampler2.read_checkpoint();
        loaded_sampler2.run();
        check_equal(sampler1, loaded_sampler2, num_threads);
    }

BOOST_AUTO_TEST_SUITE_END()
