#include <hops/FileWriter/FileWriterFactory.hpp>
#include "Fixtures.hpp"

template<typename Matrix, typename Vector, typename Model>
void sample(long seed, const std::string &filename) {
    using Polytope = PolytopeSpaceFixture<Model, Matrix, Vector>;

    constexpr const long numberOfSamples = 10000;

    Polytope polytope;
    const long thinning = 100 * polytope.polytopeSpace.roundedA.cols();
    std::cout << "thinning is " << thinning << ", " << std::flush;
    hops::MarkovChainAdapter
            markovChain(
            hops::StateRecorder(
                    hops::StateTransformation(
                            hops::NoOpDraw(
                                    hops::CoordinateHitAndRunProposal<Matrix, Vector>(
                                            polytope.polytopeSpace.roundedA,
                                            polytope.polytopeSpace.roundedb,
                                            polytope.polytopeSpace.roundedStartingPoint)),
                            hops::Transformation(polytope.polytopeSpace.roundedN, polytope.polytopeSpace.roundedShift)
                    )
            )
    );
    hops::RandomNumberGenerator randomNumberGenerator(std::random_device{}(), seed);
    // burn in
    markovChain.draw(randomNumberGenerator, 1, numberOfSamples);
    markovChain.clearHistory();

    // Sample
    long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    markovChain.draw(randomNumberGenerator, numberOfSamples, thinning);
    long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    std::cout << Model::name << " sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
              << " seconds, that's "
              << static_cast<double>(endEpoch - startEpoch) / (1000 * thinning * numberOfSamples) << " s per sample"
              << std::endl;

    // Create FileWriter
    auto fileWriter = hops::FileWriterFactory::createFileWriter(filename, hops::FileWriterType::Csv);

    // Write samples to file
    markovChain.writeHistory(fileWriter.get());
    markovChain.clearHistory();
}

template<typename Model>
void runChains() {
    std::cout << "starting " << Model::name << ", " << std::flush;
    sample<Eigen::MatrixXd, Eigen::VectorXd, Model>(
            1,
            std::string(Model::name) + "_CHRRS_d");
}

int main() {
//    runChains<simplex_64D>();
//    runChains<simplex_256D>();
//    runChains<simplex_512D>();
//    runChains<simplex_1024D>();
//    runChains<simplex_2048D>();
//    runChains<e_coli_core>();
//    runChains<iAT_PLT_636>();
//    runChains<iJO1366>();
//    runChains<RECON1>();
    runChains<Recon2>();
//    runChains<Recon3D>();
}
