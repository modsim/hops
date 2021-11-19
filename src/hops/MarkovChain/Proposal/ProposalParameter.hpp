#ifndef HOPS_PROPOSALPARAMETER_HPP
#define HOPS_PROPOSALPARAMETER_HPP


#include <string>
#include <vector>

#include <hops/Utility/VectorType.hpp>

namespace hops {
    class ProposalParameter {
    public:
        virtual ~ProposalParameter() = default;

        [[nodiscard]]  virtual VectorType getValues() const = 0;

        virtual void setValues(const VectorType &) const = 0;

        [[nodiscard]] virtual std::vector<std::string> getParameterNames() const = 0;
    };
}


#endif //HOPS_PROPOSALPARAMETER_HPP
