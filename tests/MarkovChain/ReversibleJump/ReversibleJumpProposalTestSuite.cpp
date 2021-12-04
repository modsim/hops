#define BOOST_TEST_MODULE ReversibleJumpProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <Eigen/Core>
#include <utility>
#include <vector>

#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

#include <hops/MarkovChain/Proposal/ReversibleJumpProposal.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>
#include <hops/Utility/MatrixType.hpp>
#include <hops/Utility/VectorType.hpp>

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

    class FullGammaModel {
    public:
        using FloatType = double;

        explicit FullGammaModel(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
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

        [[nodiscard]] FloatType computeNegativeLogLikelihood(const hops::VectorType &parameters) const {
            FloatType neglike = 0;
            for (const auto &measurement : this->measurements) {
                neglike -= std::log(
                        gammaProbabilityDensityFunction(measurement,
                                                        parameters(0),
                                                        parameters(1),
                                                        parameters(2)));

            }
            return neglike;
        }

        [[nodiscard]] const std::vector<std::string> &getParameterNames() const {
            return parameterNames;
        }

        hops::VectorType getB() const {
            return b;
        }

        hops::MatrixType getA() const {
            return A;
        }

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"location", "scale", "shape"};

        hops::VectorType b;
        hops::MatrixType A;
        std::vector<FloatType> measurements;
        std::string modelName = "FullGammaModel";
    };


    class GammaModel1 {
    public:
        using FloatType = double;

        explicit GammaModel1(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
            A = hops::MatrixType(4, 2);
            A << 1, 0,
                    -1, 0,
                    0, 1,
                    0, -1;

            b = hops::VectorType(4);
            b << 0.9, 0., 10, -0.1;
        }

        FloatType computeNegativeLogLikelihood(const hops::VectorType &parameters) const {
            FloatType neglike = 0;
            for (const auto &measurement : this->measurements) {
                neglike -= std::log(
                        gammaProbabilityDensityFunction(measurement,
                                                        parameters(0),
                                                        scale,
                                                        parameters(1)));

            }
            return neglike;
        }

        [[nodiscard]] const std::vector<std::string> &getParameterNames() const {
            return parameterNames;
        }

        hops::VectorType getB() const {
            return b;
        }

        hops::MatrixType getA() const {
            return A;
        }

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"location", "shape"};

        hops::VectorType b;
        hops::MatrixType A;
        std::vector<FloatType> measurements;
        constexpr static const double scale = 1;
        std::string modelName = "GammaModel1";
    };

    class GammaModel2 {
    public:
        using FloatType = double;

        explicit GammaModel2(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
            A = hops::MatrixType(4, 2);
            A << 1, 0,
                    -1, 0,
                    0, 1,
                    0, -1;

            b = hops::VectorType(4);
            b << 10, -0.1, 10, -0.1;
        }

        [[nodiscard]] FloatType computeNegativeLogLikelihood(const hops::VectorType &parameters) const {
            FloatType neglike = 0;
            for (const auto &measurement : this->measurements) {
                neglike -= std::log(
                        gammaProbabilityDensityFunction(measurement,
                                                        location,
                                                        parameters(0),
                                                        parameters(1)));

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

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"scale", "shape"};

        hops::VectorType b;
        hops::MatrixType A;
        std::vector<FloatType> measurements;

        constexpr static double location = 0;
        std::string modelName = "GammaModel2";
    };


    class MinimalGammaModel {
    public:
        using FloatType = double;

        explicit MinimalGammaModel(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
            A = hops::MatrixType(2, 1);
            A << 1, -1;
            b = hops::VectorType(2);
            b << 10, -0.1;
        }

        [[nodiscard]] FloatType computeNegativeLogLikelihood(const hops::VectorType &parameters) const {
            FloatType neglike = 0;
            for (const auto &measurement : this->measurements) {
                neglike -= std::log(
                        gammaProbabilityDensityFunction(measurement,
                                                        location,
                                                        scale,
                                                        parameters(0)));

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

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"shape"};

        hops::VectorType b;
        hops::MatrixType A;
        std::vector<FloatType> measurements;

        constexpr static double location = 0;
        constexpr static double scale = 1;
        std::string modelName = "MinimalGammaModel";
    };
}

BOOST_AUTO_TEST_SUITE(ReversibleJumpProposal)

// TODO fix to work with updated api
    BOOST_AUTO_TEST_CASE(GammaModel) {
        // The gamma model is a simple test case where the RJMCMC algorithm jumps a nested model structure of gamma functions.
        double measurement = 1;
        FullGammaModel model({measurement});
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
        long thinning = 10;
        long numberOfSamples = 100'0000;
        markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);


        std::vector<double> model_visit_counts = {0, 0, 0, 0};
        std::vector<double> location;
        std::vector<double> scale;
        std::vector<double> shape;

        for (auto state: markovChain->getStateRecords()) {
            double model_binary_name = state(0);

            int index = model_binary_name == 7 ? 0 :
                        model_binary_name == 5 ? 1 :
                        model_binary_name == 3 ? 2 : 3;

            model_visit_counts[index]++;
            location.emplace_back(state(1));
            scale.emplace_back(state(2));
            shape.emplace_back(state(3));
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
                15, // 0.15325739
                31, // 0.31479786
        };

        BOOST_ASSERT(actualModelProbabilities.size() == expectedModelProbabilityPercentages.size());
        for (size_t i = 0; i < actualModelProbabilities.size(); ++i) {
            std::cout << actualModelProbabilities[i] << std::endl;
            BOOST_CHECK_EQUAL(
                    actualModelProbabilities[i],
                    expectedModelProbabilityPercentages[i]
            );
        }
    }

BOOST_AUTO_TEST_SUITE_END()

