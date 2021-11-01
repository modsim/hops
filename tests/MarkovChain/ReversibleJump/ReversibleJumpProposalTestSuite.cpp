#define BOOST_TEST_MODULE ReversibleJumpProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/ReversibleJump/ReversibleJumpProposal.hpp>
#include <hops/Model/GammaModels.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>

BOOST_AUTO_TEST_SUITE(ReversibleJumpProposal)

    BOOST_AUTO_TEST_CASE(GammaModel) {
        // The gamma model is a simple test case where the RJMCMC algorithm jumps a nested model structure of gamma functions.
        double measurement = 1;
        hops::FullGammaModel model({measurement});
        auto A = model.getA();
        auto b = model.getB();

        Eigen::VectorXd start(3);
        start << 0.5, 0.5, 0.5;

        Eigen::VectorXi jumpIndices(2);
        // location and scale are parameters suitable for jumping. They are the first and second parameters respectively
        jumpIndices << 0, 1;
        Eigen::VectorXd defaultValues(3);
        defaultValues << 0, 1, 1;

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::StateRecorder(
                        hops::ReversibleJumpProposal(
                                hops::HitAndRunProposal(
                                        A,
                                        b,
                                        start),
                                model,
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        auto markovChain = std::make_unique<decltype(markovChainImpl)>(markovChainImpl);

        hops::RandomNumberGenerator randomNumberGenerator(42);


        for (auto p : markovChain->getParameterNames()) {
            std::cout << p << std::endl;
        }
        long thinning = 100;
        long numberOfSamples = 10'000;
        markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);


        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::vector<double> parameters = {0, 0, 0};

        for (auto state: markovChain->getStateRecords()) {
            double model_binary_name = state(0);

            int index = model_binary_name == 7 ? 0 :
                        model_binary_name == 5 ? 1 :
                        model_binary_name == 3 ? 2 : 3;

            model_visit_counts[index]++;
        }

        std::vector<int> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count:model_visit_counts) {
            actualModelProbabilities.emplace_back(count * 100 / numberOfSamples);
        }

        // Actual model probabilities integrated with scipy and rounded:
        std::vector<int> expectedModelProbabilityPercentages{
            18, // 0.18335007
            34, // 0.34859468
            16, // 0.15325739
            31, // 0.31479786
        };

        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for(long i=0; i<actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_EQUAL(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i]
            );
        }

        // TODO add checks the parameter inferences are correct
    }

BOOST_AUTO_TEST_SUITE_END()

