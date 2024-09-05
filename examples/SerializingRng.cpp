#include "hops/hops.hpp"
#include <iostream>
#include <random>

int main() {
    std::normal_distribution<double> normalDistribution(0., 1.);
    int seed = 0;
    int numbers_to_generate = 4;

    for(int i=0; i<numbers_to_generate; ++i) {
        hops::RandomNumberGenerator rng(seed, seed);
        double randomNumber = normalDistribution(rng);
        std::cout << "hops random number " << i << "=" << randomNumber << std::endl;
    }

    for(int i=0; i<numbers_to_generate; ++i) {
        std::mt19937_64 rng(seed);
        double randomNumber = normalDistribution(rng);
        std::cout << "std random number " << i << "=" << randomNumber << std::endl;
    }

    for(int i=0; i<numbers_to_generate; ++i) {
        hops::RandomNumberGenerator rng(seed, seed);
        normalDistribution.reset();
        double randomNumber = normalDistribution(rng);
        std::cout << "reset hops random number " << i << "=" << randomNumber << std::endl;
    }

    for(int i=0; i<numbers_to_generate; ++i) {
        std::mt19937_64 rng(seed);
        normalDistribution.reset();
        double randomNumber = normalDistribution(rng);
        std::cout << "reset std random number " << i << "=" << randomNumber << std::endl;
    }
}

