#define BOOST_TEST_MODULE DNest4AdapterTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/NestedSampling/DNest4Adapter.hpp>
#include <Eigen/Core>

#include <hops/MarkovChain/Draw/NoOpDrawAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Proposal/GaussianProposal.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/Model/ModelMixin.hpp>

BOOST_AUTO_TEST_SUITE(DNest4AdapterTestSuite)

    BOOST_AUTO_TEST_CASE(TestGaussianModelEvidenceIsCorrect) {

        {

            Eigen::VectorXd mean(1);
            mean << 0;
            Eigen::MatrixXd covariance(1, 1);
            covariance << 1;
            // constrain to [-5, 5]
            Eigen::MatrixXd A(2, 1);
            A << 1, -1;
            Eigen::VectorXd b(2);
            b << 5, 5;

            hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd> gaussian(mean, covariance);

            auto priorSampler = hops::NoOpDrawAdapter(hops::GaussianProposal<decltype(A), decltype(b)>(A, b, mean));
            auto posteriorSampler = hops::MetropolisHastingsFilter(
                    hops::ModelMixin(
                            hops::GaussianProposal<decltype(A), decltype(b)>(A, b, mean),
                            gaussian
                    )
            );
            auto* const constructor = new hops::DNest4AdapterConstructor(priorSampler, posteriorSampler);

            hops::DNest4Adapter<decltype(priorSampler), decltype(posteriorSampler), constructor> modelImpl;

            DNest4::Options sampler_options(5, // num_particles
                            10000, // new_level_interval
                            10000, // save_interval
                            100, // thread_steps
                            80, // max_num_levels
                            10, // lambda
                            100, // beta
                            10000 // max_num_saves
            );

            class DNest4Model : public decltype(modelImpl) {
            public:
                DNest4Model() = default;
            };

            // Create sampler
            DNest4::Sampler<decltype(modelImpl)> sampler(4, // threads
                                                     2.7182818284590451, // compression double
                                                     sampler_options,
                                                     true, // save to disk
                                                     false // adaptive
            );

            // Seed RNGs
//            sampler.initialise(42);
//
//            sampler.run();
        }
    }

BOOST_AUTO_TEST_SUITE_END()
