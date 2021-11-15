#include <Eigen/Core>
#include <Eigen/Sparse>
#include <sbml/SBMLTypes.h>
#include <sbml/packages/fbc/common/FbcExtensionTypes.h>

#include "SbmlReader.hpp"

Eigen::SparseMatrix<double> parseStoichiometry(Model *model) {
    std::map<const std::string, unsigned int> speciesIdAttributeToIndex;
    for (unsigned int i = 0; i < model->getListOfSpecies()->size(); ++i) {
        Species *species = model->getSpecies(i);
        speciesIdAttributeToIndex.insert(std::make_pair(species->getIdAttribute(), i));
    }

    std::vector<Eigen::Triplet<double>> stoichiometricTriplets;
    for (unsigned int i = 0; i < model->getListOfReactions()->size(); ++i) {
        Reaction *reaction = model->getReaction(i);
        ListOfSpeciesReferences *reactants = reaction->getListOfReactants();
        for (unsigned int j = 0; j < reactants->size(); ++j) {
            SimpleSpeciesReference *reactant = reactants->get(j);
            std::string speciesIdAttribute;
            reactant->getAttribute("species", speciesIdAttribute);
            unsigned int k = speciesIdAttributeToIndex.find(speciesIdAttribute)->second;
            double value;
            reactant->getAttribute("stoichiometry", value);
            stoichiometricTriplets.emplace_back(k, i, -value);
        }
        ListOfSpeciesReferences *products = reaction->getListOfProducts();
        for (unsigned int j = 0; j < products->size(); ++j) {
            SimpleSpeciesReference *product = products->get(j);
            std::string speciesIdAttribute;
            product->getAttribute("species", speciesIdAttribute);
            unsigned int k = speciesIdAttributeToIndex.find(speciesIdAttribute)->second;
            double value;
            product->getAttribute("stoichiometry", value);
            stoichiometricTriplets.emplace_back(k, i, value);
        }
    }
    Eigen::SparseMatrix<double> stoichiometry(model->getListOfSpecies()->size(),
                                              model->getListOfReactions()->size());
    stoichiometry.setFromTriplets(stoichiometricTriplets.begin(), stoichiometricTriplets.end());
    return stoichiometry;
}

std::tuple<Eigen::SparseMatrix<double>, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
parseConstraints(Model *model) {
    unsigned int numberOfReactions = model->getListOfReactions()->size();
    Eigen::VectorXd b(numberOfReactions * 2);
    Eigen::VectorXd lb(numberOfReactions);
    Eigen::VectorXd ub(numberOfReactions);
    std::vector<Eigen::Triplet<double>> triplets;
    for (unsigned int i = 0; i < numberOfReactions; ++i) {
        Reaction *reaction = model->getReaction(i);
        FbcReactionPlugin *fluxBalanceConstraintsPlugin = dynamic_cast<FbcReactionPlugin *>(reaction->getPlugin("fbc"));
        ub(i) = model->getParameter(fluxBalanceConstraintsPlugin->getUpperFluxBound())->getValue();
        lb(i) = model->getParameter(fluxBalanceConstraintsPlugin->getLowerFluxBound())->getValue();
        b(i) = model->getParameter(fluxBalanceConstraintsPlugin->getUpperFluxBound())->getValue();
        b(i + numberOfReactions) = model->getParameter(fluxBalanceConstraintsPlugin->getLowerFluxBound())->getValue();
        triplets.emplace_back(i, i, 1);
        triplets.emplace_back(i + numberOfReactions, i, -1);
    }

    Eigen::SparseMatrix<double> C(2 * numberOfReactions, numberOfReactions);
    C.setFromTriplets(triplets.begin(), triplets.end());
    return std::make_tuple(C, b, ub, lb);
}

template<typename MatrixType, typename VectorType>
hops::SbmlModel<MatrixType, VectorType> hops::SbmlReader::readModel(const std::string &file) {
    auto document = std::unique_ptr<SBMLDocument>(readSBML(file.c_str()));

    if (document->getNumErrors() > 0) {
        std::cerr << "Encountered the following SBML errors:" << std::endl;
        document->printErrors(std::cerr);
        throw std::runtime_error("SBML errors.");
    }

    Model *model = document->getModel();

    if (!model) {
        throw std::runtime_error("No model present.");
    }

    SbmlModel<MatrixType, VectorType> sbmlModel;

    sbmlModel.setStoichiometry(MatrixType(parseStoichiometry(model).cast<typename MatrixType::Scalar>()));

    auto constraints = parseConstraints(model);
    sbmlModel.setLowerBounds(std::get<3>(constraints).cast<typename MatrixType::Scalar>());
    sbmlModel.setUpperBounds(std::get<2>(constraints).cast<typename MatrixType::Scalar>());
    sbmlModel.setConstraintVector(std::get<1>(constraints).cast<typename MatrixType::Scalar>());
    sbmlModel.setConstraintMatrix(std::get<0>(constraints).cast<typename MatrixType::Scalar>());

    return sbmlModel;
}

template hops::SbmlModel<Eigen::MatrixXi, Eigen::VectorXi> hops::SbmlReader::readModel(const std::string &file);

template hops::SbmlModel<Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<long, Eigen::Dynamic, 1>>
hops::SbmlReader::readModel(const std::string &file);

template hops::SbmlModel<Eigen::MatrixXf, Eigen::VectorXf> hops::SbmlReader::readModel(const std::string &file);

template hops::SbmlModel<Eigen::MatrixXd, Eigen::VectorXd> hops::SbmlReader::readModel(const std::string &file);

template hops::SbmlModel<Eigen::SparseMatrix<int>, Eigen::VectorXi>
hops::SbmlReader::readModel(const std::string &file);

template hops::SbmlModel<Eigen::SparseMatrix<long>, Eigen::Matrix<long, Eigen::Dynamic, 1>>
hops::SbmlReader::readModel(const std::string &file);

template hops::SbmlModel<Eigen::SparseMatrix<float>, Eigen::VectorXf>
hops::SbmlReader::readModel(const std::string &file);

template hops::SbmlModel<Eigen::SparseMatrix<double>, Eigen::VectorXd>
hops::SbmlReader::readModel(const std::string &file);
