#ifndef HOPS_GAMMAMODELS_HPP
#define HOPS_GAMMAMODELS_HPP

#include <cmath>
#include <Eigen/Core>
#include <vector>
#include <utility>
#include <vector>

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
}

namespace hops {

    template<typename Matrix=Eigen::MatrixXd, typename Vector=Eigen::VectorXd>
    class FullGammaModel {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;
        using FloatType = typename VectorType::Scalar;

        explicit FullGammaModel(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
            A = MatrixType(6, 3);
            A << 1, 0, 0,
                    -1, 0, 0,
                    0, 1, 0,
                    0, -1, 0,
                    0, 0, 1,
                    0, 0, -1;

            b = VectorType(6);
            b << 0.9, 0., 10, -0.1, 10, -0.1;
        }

        FloatType computeNegativeLogLikelihood(const VectorType &parameters) const {
            FloatType neglike = 0;
            for (const auto &measurement : this->measurements) {
                neglike -= std::log(
                        gammaProbabilityDensityFunction(measurement,
                                                        parameters(0),
                                                        parameters(1),
                                                        parameters(2)));

            }
//            if (parameters(1) != 1){
//                neglike = INFINITY;}
            return neglike;
        }

        [[nodiscard]] const std::vector<std::string> &getParameterNames() const {
            return parameterNames;
        }

        VectorType getB() const {
            return b;
        }

        MatrixType getA() const {
            return A;
        }

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"location", "scale", "shape"};

        VectorType b;
        MatrixType A;
        std::vector<FloatType> measurements;
        std::string modelName = "FullGammaModel";
    };


    template<typename Matrix=Eigen::MatrixXd, typename Vector=Eigen::VectorXd>
    class GammaModel1 {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;
        using FloatType = typename VectorType::Scalar;

        explicit GammaModel1(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
            A = MatrixType(4, 2);
            A << 1, 0,
                    -1, 0,
                    0, 1,
                    0, -1;

            b = VectorType(4);
            b << 0.9, 0., 10, -0.1;
        }

        FloatType computeNegativeLogLikelihood(const VectorType &parameters) const {
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

        VectorType getB() const {
            return b;
        }

        MatrixType getA() const {
            return A;
        }

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"location", "shape"};

        VectorType b;
        MatrixType A;
        std::vector<FloatType> measurements;
        constexpr static const double scale = 1;
        std::string modelName = "GammaModel1";
    };

    template<typename Matrix=Eigen::MatrixXd, typename Vector=Eigen::VectorXd>
    class GammaModel2 {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;
        using FloatType = typename VectorType::Scalar;

        explicit GammaModel2(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
            A = MatrixType(4, 2);
            A << 1, 0,
                    -1, 0,
                    0, 1,
                    0, -1;

            b = VectorType(4);
            b << 10, -0.1, 10, -0.1;
        }

        FloatType computeNegativeLogLikelihood(const VectorType &parameters) const {
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

        VectorType getB() const {
            return b;
        }

        MatrixType getA() const {
            return A;
        }

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"scale", "shape"};

        VectorType b;
        MatrixType A;
        std::vector<FloatType> measurements;

        constexpr static double location = 0;
        std::string modelName = "GammaModel2";
    };


    template<typename Matrix=Eigen::MatrixXd, typename Vector=Eigen::VectorXd>
    class MinimalGammaModel {
    public:
        using MatrixType = Matrix;
        using VectorType = Vector;
        using FloatType = typename VectorType::Scalar;

        explicit MinimalGammaModel(std::vector<FloatType> measurements) : measurements(std::move(measurements)) {
            A = MatrixType(2, 1);
            A << 1, -1;
            b = VectorType(2);
            b << 10, -0.1;
        }

        FloatType computeNegativeLogLikelihood(const VectorType &parameters) const {
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

        VectorType getB() const {
            return b;
        }

        MatrixType getA() const {
            return A;
        }

        [[nodiscard]] const std::string &getModelName() const {
            return modelName;
        }

    private:
        std::vector<std::string> parameterNames = {"shape"};

        VectorType b;
        MatrixType A;
        std::vector<FloatType> measurements;

        constexpr static double location = 0;
        constexpr static double scale = 1;
        std::string modelName = "MinimalGammaModel";
    };
}

#endif //HOPS_GAMMAMODELS_HPP
