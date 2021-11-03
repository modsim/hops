#define BOOST_TEST_MODULE DNest4AdapterTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/NestedSampling/DNest4Adapter.hpp>
#include <Eigen/Core>

#include <hops/MarkovChain/Proposal/ProposalFactory.hpp>
#include <hops/Model/Gaussian.hpp>
#include <hops/Model/ModelMixin.hpp>
#include <hops/MarkovChain/Proposal/GaussianProposal.hpp>

BOOST_AUTO_TEST_SUITE(DNest4AdapterTestSuite)

    BOOST_AUTO_TEST_CASE(TestGaussianModelEvidenceIsCorrectWithCoordinateHitAndRunSampler) {
        {
            hops::VectorType mean(1);
            mean << 0;
            Eigen::MatrixXd covariance(1, 1);
            covariance << 1;
            // constrain to [-5, 5]
            Eigen::MatrixXd A(2, 1);
            A << 1, -1;
            hops::VectorType b(2);
            b << 5, 5;

            auto gaussian = std::make_shared<hops::Gaussian>(mean, covariance);
            auto proposer = hops::ProposalFactory::createProposal<decltype(A), decltype(b), hops::GaussianProposal<decltype(A), decltype(b)>>(A, b, mean);

            hops::DNest4EnvironmentSingleton::getInstance().setProposer(std::move(proposer));
            hops::DNest4EnvironmentSingleton::getInstance().setModel(gaussian);
            hops::DNest4Adapter DNest4Model;

            DNest4::Options sampler_options(5, // num_particles
                                            10000, // new_level_interval
                                            10000, // save_interval
                                            100, // thread_steps
                                            80, // max_num_levels
                                            10, // lambda
                                            100, // beta
                                            10000 // max_num_saves
            );


            // Disable output
            std::cout.setstate(std::ios_base::failbit);
            DNest4::Sampler<decltype(DNest4Model)> sampler(1, // threads
                                                           2.7182818284590451, // compression double
                                                           sampler_options,
                                                           true, // save to disk
                                                           true // adaptive
            );

            sampler.initialise(42);
            sampler.run();
            std::cout.clear();
        }
    }

BOOST_AUTO_TEST_SUITE_END()
