#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GaussianProcessTestSuite

#include <boost/test/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#include "hops/Optimization/GaussianProcess.hpp"
#include "hops/Optimization/Kernel/SquaredExponentialKernel.hpp"

BOOST_AUTO_TEST_SUITE(GaussianProcessTestSuite)

    BOOST_AUTO_TEST_CASE(TrainAndSample) {
        hops::SquaredExponentialKernel kernel = hops::SquaredExponentialKernel<Eigen::MatrixXd, Eigen::VectorXd>(1, 1);
        hops::GaussianProcess gp = hops::GaussianProcess<Eigen::MatrixXd, Eigen::VectorXd, decltype(kernel)>(kernel, 0);
        hops::RandomNumberGenerator randomNumberGenerator(3);

        Eigen::VectorXd x(5);
        x << -2, -1, 0, 1, 2;
        Eigen::VectorXd y(5);
        y << 3.5, 1.2, 0, 0.9, 4.2;
        Eigen::VectorXd error(5);
        error << 0.01, 0.1, 1, 10, 100;
        gp.addObservations(x, y, error);

        size_t N = 100;
        double a = -5, b = 5;
        Eigen::VectorXd xPrime(N);
        for (size_t i = 0; i < N; ++i) {
            xPrime(i) = (a + i * (b - a) / N);
        }

        Eigen::VectorXd sample = gp.sample(xPrime, randomNumberGenerator);

        auto posteriorMean = gp.getPosteriorMean();
        auto posteriorCovariance = gp.getPosteriorCovariance().diagonal();

        std::vector<double> expectedPosteriorMean{0.0465881, 0.0625124, 0.0830364, 0.109189, 0.142131, 0.183145,
                                                  0.23361, 0.294964, 0.368654, 0.456073, 0.558474, 0.676885, 0.812002,
                                                  0.964086, 1.13286, 1.3174, 1.51608, 1.72651, 1.9455, 2.16912, 2.39272,
                                                  2.6111, 2.81862, 3.00944, 3.17774, 3.31801, 3.42527, 3.49536, 3.52515,
                                                  3.51277, 3.45767, 3.36076, 3.22434, 3.05205, 2.84869, 2.62001,
                                                  2.37245, 2.11286, 1.84816, 1.58503, 1.32971, 1.08766, 0.863485,
                                                  0.660724, 0.481846, 0.328233, 0.200238, 0.0972899, 0.0180339,
                                                  -0.0395105, -0.0777757, -0.0994761, -0.107448, -0.10451, -0.0933446,
                                                  -0.076407, -0.0558653, -0.0335617, -0.0109998, 0.0106496, 0.0305263,
                                                  0.0480492, 0.0628774, 0.074868, 0.084035, 0.0905101, 0.0945076,
                                                  0.096294, 0.0961639, 0.0944202, 0.0913594, 0.0872619, 0.0823848,
                                                  0.0769583, 0.0711842, 0.0652358, 0.0592591, 0.0533742, 0.0476776,
                                                  0.0422448, 0.0371319, 0.0323784, 0.0280092, 0.0240366, 0.0204621,
                                                  0.0172783, 0.0144709, 0.0120196, 0.00990032, 0.0080859, 0.00654766,
                                                  0.0052563, 0.00418284, 0.00329927, 0.0025792, 0.00199818, 0.00153403,
                                                  0.00116694, 0.000879536, 0.000656777};

        std::vector<double> expectedPosteriorCovarianceDiagonal{0, 0.999672, 0.999423, 0.999004, 0.998316, 0.997212,
                                                                0.995477, 0.992812, 0.988811, 0.982941, 0.974526,
                                                                0.962746, 0.946648, 0.925182, 0.897268, 0.861892,
                                                                0.81823, 0.765797, 0.704598, 0.635263, 0.559137,
                                                                0.478299, 0.395495, 0.313958, 0.237142, 0.168368,
                                                                0.110462, 0.0653975, 0.0340427, 0.0160376, 0.00984657,
                                                                0.0129859, 0.0223974, 0.0349086, 0.0477009, 0.0587047,
                                                                0.0668503, 0.0721353, 0.0755027, 0.0785648, 0.0832343,
                                                                0.0913396, 0.104301, 0.122921, 0.147318, 0.176995,
                                                                0.211011, 0.248205, 0.287415, 0.327654, 0.368207,
                                                                0.408647, 0.448788, 0.488596, 0.528079, 0.567207,
                                                                0.605851, 0.643771, 0.680629, 0.716033, 0.749591,
                                                                0.780955, 0.80986, 0.836144, 0.859747, 0.880707,
                                                                0.899134, 0.915192, 0.929078, 0.941003, 0.951181,
                                                                0.959817, 0.967104, 0.973221, 0.978328, 0.982571,
                                                                0.986077, 0.988959, 0.991314, 0.993226, 0.994768,
                                                                0.996002, 0.99698, 0.997746, 0.998341, 0.998796,
                                                                0.999139, 0.999394, 0.99958, 0.999714, 0.999809,
                                                                0.999874, 0.999919, 0.999949, 0.999968, 0.999981,
                                                                0.999988, 0.999993, 0.999996, 0.999998};

        std::vector<double> expectedSample{0.610922, 0.64486, 0.705069, 0.787798, 0.88899, 1.0045, 1.13024, 1.26225,
                                           1.39679, 1.5304, 1.66004, 1.78324, 1.89832, 2.00455, 2.10229, 2.193, 2.27911,
                                           2.36372, 2.45019, 2.54157, 2.64002, 2.7463, 2.8594, 2.97632, 3.09213,
                                           3.20033, 3.29336, 3.36327, 3.40256, 3.4049, 3.3658, 3.28308, 3.15713,
                                           2.99092, 2.78976, 2.56086, 2.31274, 2.05459, 1.79547, 1.54376, 1.30652,
                                           1.08916, 0.895145, 0.72602, 0.58154, 0.459976, 0.358533, 0.273802, 0.202221,
                                           0.140462, 0.0857256, 0.0359089, -0.0103512, -0.0537404, -0.0944045,
                                           -0.132123, -0.166479, -0.19699, -0.223181, -0.244593, -0.260741, -0.271037,
                                           -0.274715, -0.270782, -0.258028, -0.235103, -0.200665, -0.153581, -0.0931472,
                                           -0.0192997, 0.0672363, 0.164886, 0.271279, 0.38341, 0.497909, 0.611357,
                                           0.72062, 0.823137, 0.917113, 1.00159, 1.07639, 1.14189, 1.19875, 1.24755,
                                           1.28849, 1.3211, 1.34408, 1.3553, 1.35198, 1.33095, 1.28905, 1.22362, 1.1329,
                                           1.01648, 0.875502, 0.712798, 0.532799, 0.341281, 0.144969, -0.0489564};


        for (size_t i = 0; i < N; ++i) {
            BOOST_CHECK_SMALL(posteriorMean(i) - expectedPosteriorMean[i], 1.e-5);
        }

//        for (size_t i = 0; i < N; ++i) {
//            BOOST_CHECK_SMALL(posteriorCovariance(i) - expectedPosteriorCovarianceDiagonal[i], 1.e-5);
//        }

        for (size_t i = 0; i < N; ++i) {
            BOOST_CHECK_SMALL(sample[i] - expectedSample[i], 1.e-5);
        }
    }

    BOOST_AUTO_TEST_CASE(SquaredExponentialKernel) {
        hops::SquaredExponentialKernel kernel = hops::SquaredExponentialKernel<Eigen::MatrixXd, Eigen::VectorXd>(1, 1);
        Eigen::VectorXd x(2);
        x << 0, 1;

        auto actualCovariance = kernel(x, x);

        Eigen::MatrixXd expectedCovariance = Eigen::MatrixXd(2, 2);
        expectedCovariance << 1., 0.60653066, 0.60653066, 1.;

        for (long i = 0; i < actualCovariance.size(); ++i) {
            BOOST_CHECK_SMALL(actualCovariance(i) - expectedCovariance(i), 1.e-7);
        }
    }

BOOST_AUTO_TEST_SUITE_END()

