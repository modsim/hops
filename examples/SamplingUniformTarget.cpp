#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/StateTransformation.hpp>
#include <hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <hops/Transformation/Transformation.hpp>
#include <hops/MarkovChain/Draw/NoOpDrawAdapter.hpp>

/**
 * @details Samples a model uniformly using CHRR.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "Usage: SamplingUniformTarget model_name numberOfSamples ThinningFactor random_seed" << std::endl
                  << "Output will be stored in the directory ${model_name}${random_seed}" << std::endl
                  << "Example: SamplingUniformTarget e_coli_core 42" << std::endl << std::endl
                  << "Note that in this example the files have to be in the current working directory" << std::endl
                  << "A_e_coli_core_rounded.csv\twhich is the A from Ax<b" << std::endl
                  << "b_e_coli_core_rounded.csv\twhich is the b from Ax<b" << std::endl
                  << "start_e_coli_core_rounded.csv\twhich is an interior point of Ax<b" << std::endl
                  << "N_e_coli_core_rounded.csv\twhich is the unrounding transformation x = Ny+shift" << std::endl
                  << "p_shift_e_coli_core_rounded.csv\twhich is the shift for the unrounding transformation x = Ny+shift"
                  << std::endl << std::endl
                  << "Arguments:" << std::endl
                  << "model_name\t\t" << "name of model" << std::endl
                  << "numberOfSamples\t\t" << "how many samples to draw" << std::endl
                  << "thinningFactor\t\t" << "the total thinning is the number of dimensions times the thinning factor"
                  << std::endl
                  << "random_seed\t\t" << "seed for rng" << std::endl;
        exit(0);
    }
    std::string modelName = std::string(argv[1]);
    std::string Afile = "A_" + std::string(argv[1]) + "_rounded.csv";
    std::string bfile = "b_" + std::string(argv[1]) + "_rounded.csv";
    std::string startfile = "start_" + std::string(argv[1]) + "_rounded.csv";
    std::string Nfile = "N_" + std::string(argv[1]) + "_rounded.csv";
    std::string p_shiftfile = "p_shift_" + std::string(argv[1]) + "_rounded.csv";
    long numberOfSamples = std::stol(argv[2]);
    long thinningFactor = std::stol(argv[3]);
    std::string stringSeed = std::string(argv[4]);


    std::cout << "Sampling model: " << modelName << std::endl;
    std::cout << "prerounded A: " << Afile << std::endl;
    std::cout << "prerounded b: " << bfile << std::endl;
    std::cout << "prerounded start: " << startfile << std::endl;
    std::cout << "prerounded N: " << Nfile << std::endl;
    std::cout << "prerounded p_shift: " << p_shiftfile << std::endl;
    std::cout << "seed: " << stringSeed << std::endl;
    auto A = hops::CsvReader::readMatrix<Eigen::MatrixXd>(Afile);
    auto N = hops::CsvReader::readMatrix<Eigen::MatrixXd>(Nfile);
    auto b = hops::CsvReader::readVector<Eigen::VectorXd>(bfile);
    auto s = hops::CsvReader::readVector<Eigen::VectorXd>(startfile);
    auto shift = hops::CsvReader::readVector<Eigen::VectorXd>(p_shiftfile);
    int seed = std::stoi(stringSeed);

    hops::RandomNumberGenerator randomNumberGenerator(seed);

    auto markovChain = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::StateRecorder(
                            hops::StateTransformation(
                                    hops::CoordinateHitAndRunProposal(
                                            A,
                                            b,
                                            s),
                                    hops::Transformation(N, shift))
                    )
            )
    );


    markovChain.draw(randomNumberGenerator, numberOfSamples, A.cols() * thinningFactor);
    markovChain.writeHistory(
            hops::FileWriterFactory::createFileWriter(std::string(argv[1]) + stringSeed,
                                                      hops::FileWriterType::Csv).get());
    markovChain.clearHistory();
}
