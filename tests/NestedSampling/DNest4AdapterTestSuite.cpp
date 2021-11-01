//#define BOOST_TEST_MODULE DNest4AdapterTestSuite
//#define BOOST_TEST_DYN_LINK
//
//#include <boost/test/included/unit_test.hpp>
//#include <hops/NestedSampling/DNest4Adapter.hpp>
//#include <Eigen/Core>
//
//#include <hops/MarkovChain/Draw/NoOpDrawAdapter.hpp>
//#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
//#include <hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
//#include <hops/Model/MultivariateGaussian.hpp>
//#include <hops/Model/ModelMixin.hpp>
//
//BOOST_AUTO_TEST_SUITE(DNest4AdapterTestSuite)
//
//    BOOST_AUTO_TEST_CASE(TestGaussianModelEvidenceIsCorrectWithCoordinateHitAndRunSampler) {
//        {
//            class ModelImpl {
//            public:
//                typedef Eigen::VectorXd StateType;
//
//                ~ModelImpl() {
////                    delete priorSampler;
////                    delete posteriorSampler;
//                }
//
//                ModelImpl() {
//                    StateType mean(1);
//                    mean << 0;
//                    Eigen::MatrixXd covariance(1, 1);
//                    covariance << 1;
//                    // constrain to [-5, 5]
//                    Eigen::MatrixXd A(2, 1);
//                    A << 1, -1;
//                    StateType b(2);
//                    b << 5, 5;
//                    hops::MultivariateGaussian<Eigen::MatrixXd, StateType> gaussian(mean, covariance);
//                    model = gaussian;
//
//                    sampler = new hops::NoOpDrawAdapter(
//                            hops::CoordinateHitAndRunProposal<decltype(A), decltype(b)>(A, b, mean));
//                    hops::RandomNumberGenerator temp_rng((std::random_device()()));
//                    for(int i=0; i<100; ++i) {
//                        sampler->draw(temp_rng);
//                    }
//                }
//
//                hops::NoOpDrawAdapter<hops::CoordinateHitAndRunProposal<Eigen::MatrixXd, Eigen::VectorXd>>
//                *getSampler() const {
//                    return sampler;
//                }
//
//                const hops::MultivariateGaussian<Eigen::MatrixXd, StateType> &getModel() const {
//                    return model;
//                }
//
//                void seedRng(int seed) {
//                    rngInitialized = true;
//                    randomNumberGenerator.seed(seed);
//                }
//
//                [[nodiscard]] bool isRngInitialized() const {
//                    return rngInitialized;
//                }
//
//                hops::RandomNumberGenerator &getRandomNumberGenerator() {
//                    return randomNumberGenerator;
//                }
//
//            private:
//                bool rngInitialized = false;
//                hops::RandomNumberGenerator randomNumberGenerator;
//                hops::NoOpDrawAdapter<hops::CoordinateHitAndRunProposal<Eigen::MatrixXd, Eigen::VectorXd>> *sampler;
//                hops::MultivariateGaussian<Eigen::MatrixXd, StateType> model;
//            };
//
////            hops::DNest4Environment
//            hops::DNest4Adapter<ModelImpl> DNest4Model;
//
//            DNest4::Options sampler_options(5, // num_particles
//                                            10'000, // new_level_interval
//                                            1000, // save_interval
//                                            100, // thread_steps
//                                            20, // max_num_levels
//                                            5, // lambda
//                                            100, // beta
//                                            100'000 // max_num_saves
//            );
//
//
//            DNest4::Sampler<decltype(DNest4Model)> sampler(4, // threads
//                                                           2.7182818284590451, // compression double
//                                                           sampler_options,
//                                                           true, // save to disk
//                                                           true // adaptive
//            );
//
//            sampler.initialise(42);
//            sampler.run();
//        }
//    }
//
////    BOOST_AUTO_TEST_CASE(TestGaussianModelEvidenceIsCorrect) {
////        {
////            class ModelImpl {
////            public:
////                ModelImpl() {
////                    Eigen::VectorXd mean(1);
////                    mean << 0;
////                    Eigen::MatrixXd covariance(1, 1);
////                    covariance << 1;
////                    // constrain to [-5, 5]
////                    Eigen::MatrixXd A(2, 1);
////                    A << 1, -1;
////                    Eigen::VectorXd b(2);
////                    b << 5, 5;
////                    hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd> gaussian(mean, covariance);
////
////                    priorSampler = std::unique_ptr(new hops::NoOpDrawAdapter(
////                            hops::GaussianProposal<decltype(A), decltype(b)>(A, b, mean)));
////                    posteriorSampler = std::unique_ptr<decltype(*posteriorSampler)>(new hops::MetropolisHastingsFilter(
////                            hops::ModelMixin(
////                                    hops::GaussianProposal<decltype(A), decltype(b)>(A, b, mean),
////                                    gaussian
////                            )
////                                                                                )
////                    );
////                }
////
////                hops::NoOpDrawAdapter<hops::GaussianProposal<Eigen::MatrixXd, Eigen::VectorXd>>
////                getPriorSampler() const {
////                    return *priorSampler;
////                }
////
////                hops::MetropolisHastingsFilter<hops::ModelMixin<hops::GaussianProposal<Eigen::MatrixXd, Eigen::VectorXd>, hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>>>
////                getPosteriorSampler() const {
////                    return *posteriorSampler;
////                }
////
////            private:
////                std::unique_ptr<hops::NoOpDrawAdapter<hops::GaussianProposal<Eigen::MatrixXd, Eigen::VectorXd>>> priorSampler;
////                std::unique_ptr<hops::MetropolisHastingsFilter<hops::ModelMixin<hops::GaussianProposal<Eigen::MatrixXd, Eigen::VectorXd>, hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>>>> posteriorSampler;
////            };
////
////            hops::DNest4Adapter<ModelImpl> DNest4Model;
////
////            DNest4::Options sampler_options(5, // num_particles
////                                            10000, // new_level_interval
////                                            10000, // save_interval
////                                            100, // thread_steps
////                                            80, // max_num_levels
////                                            10, // lambda
////                                            100, // beta
////                                            10000 // max_num_saves
////            );
////
////
////            DNest4::Sampler<decltype(DNest4Model)> sampler(4, // threads
////                                                           2.7182818284590451, // compression double
////                                                           sampler_options,
////                                                           true, // save to disk
////                                                           false // adaptive
////            );
////
////            // Seed RNGs
//////            sampler.initialise(42);
//////
//////            sampler.run();
////        }
////    }
//
//BOOST_AUTO_TEST_SUITE_END()
