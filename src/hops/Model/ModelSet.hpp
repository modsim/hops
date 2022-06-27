#ifndef HOPS_MODELSET_HPP
#define HOPS_MODELSET_HPP

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Model.hpp"


namespace hops {
    class ModelSet {
    public:
        double computeNegativeLogLikelihood(const VectorType &x, const std::vector<unsigned char> &activeParameters);

        /**
         * @Brief For uniform prior it returns 1./V, where V is the Volume of the parameter space (convex polytope).
         * @param x
         * @param activeParameters
         * @return
         */
        double modelPriorProbability(const VectorType &x, const std::vector<unsigned char> &activeParameters);

        /**
         * @brief Returns the probability of parameter with index paramterIndex being active given activeParameters
         * @param parameterIndex
         * @param activeParameters
         * @return
         */
        double conditionalProbability(long parameterIndex, std::vector<unsigned char> &activeParameters);

        [[nodiscard]] std::vector<std::string> getDimensionNames() const;

        [[nodiscard]] std::unique_ptr<Model> copyModel() const;


    private:
        std::vector<std::string> jumpableParameters;
        std::vector<VectorType::Scalar> defaultValues;
        // alphabetical
        std::vector<std::string> allParameterNames;

        std::vector<std::unique_ptr<Model>> densities;
    };
}


#endif //HOPS_MODELSET_HPP
