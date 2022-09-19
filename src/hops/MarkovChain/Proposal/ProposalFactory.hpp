#ifndef HOPS_PROPOSALFACTORY_HPP
#define HOPS_PROPOSALFACTORY_HPP

#include <Eigen/SparseCore>

#include "hops/Model/Model.hpp"
#include "hops/Utility/VectorType.hpp"

namespace hops {
    /**
     * @details Implementation detail Proposal Factory is an abstract factory, see
     * https://refactoring.guru/design-patterns/abstract-factory/cpp/example
     */
    class ProposalFactory {
    public:
        ~ProposalFactory() = default;

        /**
         * @brief Creates model and likelihood unaware proposal.
         * @param inequalityLhs
         * @param inequalityRhs
         * @param startingPoint
         * @return
         */
        virtual std::unique_ptr<Proposal> createProposal(Eigen::MatrixXd inequalityLhs,
                                                         Eigen::VectorXd inequalityRhs,
                                                         VectorType startingPoint) = 0;

        /**
         * @brief Creates model and likelihood unaware proposal.
         * @param inequalityLhs
         * @param inequalityRhs
         * @param startingPoint
         * @return
         */
        virtual std::unique_ptr<Proposal> createProposal(Eigen::SparseMatrix<double> inequalityLhs,
                                                         Eigen::VectorXd inequalityRhs,
                                                         VectorType startingPoint) = 0;

        /**
         * @brief Creates model aware proposal. These proposals might use any of the functions
         * of the model interface.
         * @param model the factory transfers ownership of the model to the proposal. This is enforced by
         * passing the model as unique_ptr.
         * @param inequalityLhs
         * @param inequalityRhs
         * @param startingPoint
         * @return
         */
        virtual std::unique_ptr<Proposal> createProposal(std::unique_ptr<Model> model,
                                                         Eigen::MatrixXd inequalityLhs,
                                                         Eigen::VectorXd inequalityRhs,
                                                         VectorType startingPoint) = 0;

        /**
         * @brief Creates model aware proposal. These proposals might use any of the functions
         * of the model interface.
         * @param model the factory transfers ownership of the model to the proposal. This is enforced by
         * passing the model as unique_ptr.
         * @param inequalityLhs
         * @param inequalityRhs
         * @param startingPoint
         * @return
         */
        virtual std::unique_ptr<Proposal> createProposal(std::unique_ptr<Model> model,
                                                         Eigen::SparseMatrix<double> inequalityLhs,
                                                         Eigen::VectorXd inequalityRhs,
                                                         VectorType startingPoint) = 0;
    };
}

#endif //HOPS_PROPOSALFACTORY_HPP
