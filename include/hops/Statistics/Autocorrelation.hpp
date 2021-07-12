#ifndef HOPS_AUTOCORRELATION_HPP
#define HOPS_AUTOCORRELATION_HPP

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <memory>
        
namespace hops {
    inline size_t nextGoodSizeFFT(size_t N) {
        if (N <= 2) {
            return 2;
        }
        while (true) {
            size_t m = N;
            while ((m % 2) == 0) {
            m /= 2;
            }
            while ((m % 3) == 0) {
            m /= 3;
            }
            while ((m % 5) == 0) {
            m /= 5;
            }
            if (m <= 1) {
            return N;
            }
            N++;
        }
    }
       
    template <typename StateType>
    void computeAutocorrelations (const std::vector<StateType>& draws, 
                                  Eigen::VectorXd& autocorrelations, 
                                  unsigned long dimension) {
        computeAutocorrelations(&draws, autocorrelations, dimension);
    }
    
    template <typename StateType>
    void computeAutocorrelations (const std::vector<StateType>* draws, 
                                  Eigen::VectorXd& autocorrelations, 
                                  unsigned long dimension) {
        size_t N = draws->size();
        Eigen::VectorXd X = Eigen::VectorXd::Zero(N);
        for (size_t n = 0; n < N; ++n) {
            X(n) = (*draws)[n](dimension);
        }

        Eigen::FFT<typename StateType::Scalar> fft;
        size_t M = nextGoodSizeFFT(N);
        size_t Mt2 = 2 * M;

        // center and pad X
        Eigen::VectorXd centeredX(Mt2);
        centeredX.setZero();
        centeredX.head(N) = X.array() - X.mean();

        // See https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation for a quick
        // explanation on what follows
        Eigen::VectorXcd frequency(Mt2);
        fft.fwd(frequency, centeredX);
        
        frequency = frequency.cwiseAbs2();

        Eigen::VectorXcd autocorrelationsTmp(Mt2);
        fft.inv(autocorrelationsTmp, frequency);

        // use "biased" estimate as recommended by Geyer (1992)
        autocorrelations = autocorrelationsTmp.head(N).real().array() / (N * N * 2);
        autocorrelations /= autocorrelations(0);
    }
}

#endif // HOPS_AUTOCORRELATION_HPP
