#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ReversibleJumpProposalTestSuite

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <Eigen/Core>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hops/MarkovChain/MarkovChainAdapter.hpp"
#include "hops/MarkovChain/ModelMixin.hpp"
#include "hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp"
#include "hops/MarkovChain/Proposal/HitAndRunProposal.hpp"
#include "hops/MarkovChain/Proposal/ReversibleJumpProposal.hpp"
#include "hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp"
#include "hops/Model/JumpableModel.hpp"
#include "hops/Model/Model.hpp"
#include "hops/Utility/MatrixType.hpp"
#include "hops/Utility/VectorType.hpp"

namespace {
    double gammaProbabilityDensityFunction(double x, double location, double scale, double shape) {
        if (scale <= 0 || shape <= 0) {
            throw std::runtime_error("scale and shape parameters have to be larger than 0.");
        }
        if (x - location < 0) {
            return 0;
        }
        return (std::pow(x - location, shape - 1) * std::exp(-(x - location) / scale)) /
               (std::tgamma(shape) * std::pow(scale, shape));
    }

    /**
     * @brief The GammaModel is a simple nested model for unit testing RJMCMC
     */
    class GammaModel : public hops::Model {
    public:
        explicit GammaModel(std::vector<double> measurements) : measurements(std::move(measurements)) {
            A = hops::MatrixType(6, 3);
            A << 1, 0, 0,
                    -1, 0, 0,
                    0, 1, 0,
                    0, -1, 0,
                    0, 0, 1,
                    0, 0, -1;

            b = hops::VectorType(6);
            b << 0.9, 0., 10, -0.1, 10, -0.1;
        }

        [[nodiscard]] double computeNegativeLogLikelihood(const hops::VectorType &parameters) override {
            double neglike = 0;
            for (const auto &measurement: this->measurements) {
                neglike -= std::log(gammaProbabilityDensityFunction(measurement,
                                                                    parameters(0),
                                                                    parameters(1),
                                                                    parameters(2)));

            }
            return neglike;
        }

        [[nodiscard]] const std::vector<std::string> &getParameterNames() const {
            return parameterNames;
        }

        [[nodiscard]] hops::VectorType getB() const {
            return b;
        }

        [[nodiscard]] hops::MatrixType getA() const {
            return A;
        }

        [[nodiscard]] std::vector<std::string> getDimensionNames() const override {
            return parameterNames;
        }

        [[nodiscard]] std::unique_ptr<Model> copyModel() const override {
            return std::make_unique<GammaModel>(*this);
        }

    private:
        std::vector<std::string> parameterNames = {"location", "scale", "shape"};
        hops::VectorType b;
        hops::MatrixType A;
        std::vector<double> measurements;
    };
}

struct ReversibleJumpProposalTestFixture {
public:
    ReversibleJumpProposalTestFixture() {
        start = 0.5 * Eigen::VectorXd::Ones(3);

        jumpIndices = Eigen::VectorXi(2);
        // location and scale are parameters suitable for jumping. They are the first and second parameters respectively
        jumpIndices << 0, 1;
        defaultValues = Eigen::VectorXd(2);
        // 2 default values for the jumpable parameters
        defaultValues << 0, 1;
        wrongDefaultValues = Eigen::VectorXd(3);
        // 3 default values to test wrong initialization
        wrongDefaultValues << 0, 1, 1;

        A = model.getA();
        b = model.getB();
    }

    Eigen::VectorXi jumpIndices;
    Eigen::VectorXd defaultValues;
    Eigen::VectorXd wrongDefaultValues;
    Eigen::VectorXd start;
    static constexpr double measurement = 1.;
    GammaModel model = GammaModel({measurement});
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    // expected model probabilities integrated with scipy, see resources/test_rjmcmc.py
    std::vector<double> expectedModelProbabilityPercentages{
            0.3147978599901968,
            0.15325738646688555,
            0.3485946848672095,
            0.18335006867570805,
    };

    // expected means, also found in scipy, see resources/test_rjmcmc.py
    double expectedLocationMean = 0.25526276132468817;
    double expectedScaleMean = 1.5716092072465253;
    double expectedShapeMean = 1.7841099019665436;
};

BOOST_FIXTURE_TEST_SUITE(ReversibleJumpProposal, ReversibleJumpProposalTestFixture)

    BOOST_AUTO_TEST_CASE(DimensionsMissMatch) {
        BOOST_CHECK_THROW(
                hops::ReversibleJumpProposal(nullptr,
                                             jumpIndices,
                                             wrongDefaultValues
                ),
                std::runtime_error
        );
    }

    BOOST_AUTO_TEST_CASE(ProposalName) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>, true>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto RJMCMCProposal = hops::ReversibleJumpProposal(
                std::move(proposal),
                jumpIndices,
                defaultValues);
        BOOST_CHECK_EQUAL(RJMCMCProposal.getProposalName(), "RJMCMC(HitAndRun + mixed in Model)");
    }


    BOOST_AUTO_TEST_CASE(ParameterNames) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>, true>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto RJMCMCProposal = hops::ReversibleJumpProposal(
                std::move(proposal),
                jumpIndices,
                defaultValues);

        std::vector<std::string> expectedParameterNames = {
                "step_size",
                "model_jump_probability",
                "activation_probability",
                "deactivation_probability"};

        auto actualParameterNames = RJMCMCProposal.getParameterNames();
        BOOST_CHECK_EQUAL(actualParameterNames.size(), expectedParameterNames.size());
        for (long i = 0; i < actualParameterNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualParameterNames[i], expectedParameterNames[i]);
        }
    }

    BOOST_AUTO_TEST_CASE(DimensionNames) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::UniformStepDistribution<double>, true>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto RJMCMCProposal = hops::ReversibleJumpProposal(
                std::move(proposal),
                jumpIndices,
                defaultValues);

        std::vector<std::string> expectedDimensionNames = {
                "location_activation",
                "scale_activation",
                "shape_activation",
                "location",
                "scale",
                "shape"};

        auto actualDimensionNames = RJMCMCProposal.getDimensionNames();
        BOOST_CHECK(actualDimensionNames.size() == expectedDimensionNames.size());
        for (long i = 0; i < actualDimensionNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualDimensionNames[i], expectedDimensionNames[i]);
        }

        expectedDimensionNames = std::vector<std::string>{"location_2_activation",
                                                          "scale_2_activation",
                                                          "shape_2_activation",
                                                          "location_2",
                                                          "scale_2",
                                                          "shape_2"};

        // names for activation states are added automatically, only have to rename location, scale and shape parameter.
        RJMCMCProposal.setDimensionNames({"location_2", "scale_2", "shape_2"});
        actualDimensionNames = RJMCMCProposal.getDimensionNames();
        BOOST_CHECK_EQUAL(actualDimensionNames.size(), expectedDimensionNames.size());
        for (long i = 0; i < actualDimensionNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualDimensionNames[i], expectedDimensionNames[i]);
        }
    }

    BOOST_AUTO_TEST_CASE(SetBadParameters) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::UniformStepDistribution<double>, true>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto RJMCMCProposal = hops::ReversibleJumpProposal(
                std::move(proposal),
                jumpIndices,
                defaultValues);


        BOOST_CHECK_THROW(RJMCMCProposal.setParameter(hops::ProposalParameter::MODEL_JUMP_PROBABILITY, 1.),
                          std::invalid_argument);
        BOOST_CHECK_THROW(RJMCMCProposal.setParameter(hops::ProposalParameter::MODEL_JUMP_PROBABILITY, 1.1),
                          std::invalid_argument);
        BOOST_CHECK_THROW(RJMCMCProposal.setParameter(hops::ProposalParameter::ACTIVATION_PROBABILITY, 1.),
                          std::invalid_argument);
        BOOST_CHECK_THROW(RJMCMCProposal.setParameter(hops::ProposalParameter::ACTIVATION_PROBABILITY, 1.1),
                          std::invalid_argument);
        BOOST_CHECK_THROW(RJMCMCProposal.setParameter(hops::ProposalParameter::DEACTIVATION_PROBABILITY, 1.),
                          std::invalid_argument);
        BOOST_CHECK_THROW(RJMCMCProposal.setParameter(hops::ProposalParameter::DEACTIVATION_PROBABILITY, 1.1),
                          std::invalid_argument);
    }


    BOOST_AUTO_TEST_CASE(GetAndSetParameters) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>, false>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto RJMCMCProposal = hops::ReversibleJumpProposal(
                std::move(proposal),
                jumpIndices,
                defaultValues);

        double modelJumpProbability = 0.5;
        double activationProbability = 0.5;
        double deactivationProbability = 0.5;
        double stepSize = 0.5;
        RJMCMCProposal.setParameter(hops::ProposalParameter::MODEL_JUMP_PROBABILITY, modelJumpProbability);
        RJMCMCProposal.setParameter(hops::ProposalParameter::ACTIVATION_PROBABILITY, activationProbability);
        RJMCMCProposal.setParameter(hops::ProposalParameter::DEACTIVATION_PROBABILITY, deactivationProbability);
        RJMCMCProposal.setParameter(hops::ProposalParameter::STEP_SIZE, stepSize);

        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::MODEL_JUMP_PROBABILITY)),
                modelJumpProbability);
        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::ACTIVATION_PROBABILITY)),
                activationProbability);
        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::DEACTIVATION_PROBABILITY)),
                deactivationProbability);
        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::STEP_SIZE)),
                stepSize);

        modelJumpProbability = 0.75;
        activationProbability = 0.75;
        deactivationProbability = 0.75;
        stepSize = 0.75;

        RJMCMCProposal.setParameter(hops::ProposalParameter::MODEL_JUMP_PROBABILITY, modelJumpProbability);
        RJMCMCProposal.setParameter(hops::ProposalParameter::ACTIVATION_PROBABILITY, activationProbability);
        RJMCMCProposal.setParameter(hops::ProposalParameter::DEACTIVATION_PROBABILITY, deactivationProbability);
        RJMCMCProposal.setParameter(hops::ProposalParameter::STEP_SIZE, stepSize);

        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::MODEL_JUMP_PROBABILITY)),
                modelJumpProbability);
        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::ACTIVATION_PROBABILITY)),
                activationProbability);
        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::DEACTIVATION_PROBABILITY)),
                deactivationProbability);
        BOOST_CHECK_EQUAL(
                std::any_cast<double>(RJMCMCProposal.getParameter(hops::ProposalParameter::STEP_SIZE)),
                stepSize);

    }

    BOOST_AUTO_TEST_CASE(GammaModelHR) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ReversibleJumpProposal(
                                std::move(proposal),
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }


    BOOST_AUTO_TEST_CASE(GammaModelHRModelOutsideOfRJMCMC) {
        auto proposalImpl = hops::HitAndRunProposal(
                A,
                b,
                start);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ModelMixin(
                                hops::ReversibleJumpProposal(
                                        std::move(proposal),
                                        jumpIndices,
                                        defaultValues
                                ),
                                hops::JumpableModel(model)
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }

    BOOST_AUTO_TEST_CASE(GammaModelAdaptedJumpParametersHR) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ReversibleJumpProposal(
                                std::move(proposal),
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        markovChain->setParameter(hops::ProposalParameter::MODEL_JUMP_PROBABILITY, .5);
        markovChain->setParameter(hops::ProposalParameter::ACTIVATION_PROBABILITY, .35);
        markovChain->setParameter(hops::ProposalParameter::DEACTIVATION_PROBABILITY, .25);

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }

    BOOST_AUTO_TEST_CASE(GammaModelPreciceHR) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::UniformStepDistribution<double>, true>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ReversibleJumpProposal(
                                std::move(proposal),
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }

    BOOST_AUTO_TEST_CASE(GammaModelGaussianHR) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>, false>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ReversibleJumpProposal(
                                std::move(proposal),
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }

    BOOST_AUTO_TEST_CASE(GammaModelPreciceGaussianHR) {
        auto proposalImpl = hops::ModelMixin(
                hops::HitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>, true>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ReversibleJumpProposal(
                                std::move(proposal),
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }

    BOOST_AUTO_TEST_CASE(GammaModelCHR) {
        auto proposalImpl = hops::ModelMixin(
                hops::CoordinateHitAndRunProposal(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ReversibleJumpProposal(
                                std::move(proposal),
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }

    BOOST_AUTO_TEST_CASE(GammaModelGaussianCHR) {
        auto proposalImpl = hops::ModelMixin(
                hops::CoordinateHitAndRunProposal<decltype(A), decltype(b), hops::GaussianStepDistribution<double>>(
                        A,
                        b,
                        start),
                model);

        std::unique_ptr<hops::Proposal> proposal = std::make_unique<decltype(proposalImpl)>(proposalImpl);

        auto markovChainImpl = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::ReversibleJumpProposal(
                                std::move(proposal),
                                jumpIndices,
                                defaultValues
                        )
                )
        );

        std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(
                markovChainImpl
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);

        long numberOfSamples = 2'500'000;
        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::unordered_set<std::string> model_names;
        std::vector<Eigen::VectorXd> models;
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        std::string model_name;
        double acceptanceRates = 0;
        for (long i = 0; i < numberOfSamples; ++i) {
            auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator);
            acceptanceRates += acceptanceRate;
            models.emplace_back(state.topRows(state.rows() / 2));
            int model_index = 0;

            for (long j = 0; j < jumpIndices.rows(); ++j) {
                model_index += std::pow(2, jumpIndices.rows() - 1 - j) * state(jumpIndices(j));
                model_name += std::to_string(static_cast<int>(state(jumpIndices(j))));
            }
            model_names.insert(model_name);
            model_name = "";
            model_visit_counts[model_index]++;

            location.emplace_back(state(state.rows() / 2));
            scale.emplace_back(state(state.rows() / 2 + 1));
            shape.emplace_back(state(state.rows() / 2 + 2));
        }

        double actualLocationMean = std::accumulate(location.begin(), location.end(), 0.) / location.size();
        double actualScaleMean = std::accumulate(scale.begin(), scale.end(), 0.) / scale.size();
        double actualShapeMean = std::accumulate(shape.begin(), shape.end(), 0.) / shape.size();

        std::vector<double> actualModelProbabilities;
        actualModelProbabilities.reserve(model_visit_counts.size());
        for (auto count: model_visit_counts) {
            actualModelProbabilities.emplace_back(static_cast<double>(count) / numberOfSamples);
        }

        double relative_tolerance = 10;
        BOOST_CHECK_CLOSE(actualLocationMean, expectedLocationMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualScaleMean, expectedScaleMean, relative_tolerance);
        BOOST_CHECK_CLOSE(actualShapeMean, expectedShapeMean, relative_tolerance);
        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            BOOST_CHECK_CLOSE(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i],
                    relative_tolerance
            );
        }
    }

BOOST_AUTO_TEST_SUITE_END()
