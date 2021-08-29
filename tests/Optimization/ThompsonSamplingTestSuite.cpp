#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#define BOOST_TEST_MODULE GaussianProcessTestSuite
#define BOOST_TEST_DYN_LINK

#include <unordered_map>
#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <hops/Optimization/ThompsonSampling.hpp>
#include <hops/Optimization/Kernel/SquaredExponentialKernel.hpp>

struct TestTarget : public hops::internal::ThompsonSamplingTarget<double, Eigen::VectorXd> {
	std::unordered_map<double, std::vector<double>> values;
	std::unordered_map<double, std::vector<double>> errors;
    std::vector<unsigned> count;

    TestTarget() {
        this->values = std::unordered_map<double, std::vector<double>>{
            //{0., std::vector<std::tuple<double, double>>({
            //        std::tuple<double, double>({2.5, .5}), 
            //        std::tuple<double, double>({3.5, .5}), 
            //})},
            //{1., std::vector<std::tuple<double, double>>({
            //        std::tuple<double, double>({1.5, .5})
            //})},
            {0., {2.5, 4.5}},
            {1., {1.5, 1.5}}
        };

        this->errors = std::unordered_map<double, std::vector<double>>{
            {0., {.5, .5}},
            {1., {.5, .5}}
        };

        count = std::vector<unsigned>(2, 0);
    };

	virtual std::tuple<double, double> operator()(const Eigen::VectorXd& x) override {
        auto value = values[x(0)][count[x(0)]];
        auto error = errors[x(0)][count[x(0)]];

        count[x(0)]++;
        count[x(0)] = count[x(0)] % 2;

        return {value, error};
	}
};

BOOST_AUTO_TEST_SUITE(ThompsonSamplingTestSuite)

    BOOST_AUTO_TEST_CASE(optimize) {
        using Kernel = hops::SquaredExponentialKernel<Eigen::MatrixXd, Eigen::VectorXd>;
        using GP = hops::GaussianProcess<Eigen::MatrixXd, Eigen::VectorXd, Kernel>;

        Eigen::VectorXd expectedInputs(2);
        Eigen::VectorXd expectedValues(2);
        Eigen::VectorXd expectedErrors(2);

        expectedInputs << 1, 0;
        expectedValues << 1.5, 3.5;
        expectedErrors << .5, 1.25;

        Eigen::VectorXd grid(2);
        grid << 0, 1;

        GP gp = GP(Kernel(1, 1));

        auto target = std::make_shared<TestTarget>(TestTarget());

        hops::RandomNumberGenerator rng(1);
        size_t foo;

        hops::ThompsonSampling<Eigen::MatrixXd, Eigen::VectorXd, GP>::optimize(
            3, 1, 3, gp, target, grid, rng, &foo, 0);

        
        for (long i = 0; i < grid.size(); ++i) {
            BOOST_CHECK_SMALL(expectedInputs(i) - gp.getObservedInputs()(i), 1.e-7);
            BOOST_CHECK_SMALL(expectedValues(i) - gp.getObservedValues()(i), 1.e-7);
            BOOST_CHECK_SMALL(expectedErrors(i) - gp.getObservedValueErrors()(i), 1.e-7);
        }

    }

BOOST_AUTO_TEST_SUITE_END()

