#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/StateTransformation.hpp>
#include <hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <hops/Transformation/Transformation.hpp>
#include <hops/MarkovChain/Draw/NoOpDrawAdapter.hpp>

int main(int argc, char **argv) {
    if(argc!= 3) {
        throw std::runtime_error("Requires model name and random seed as argument.");
    }
    std::string modelName = std::string(argv[1]);
    std::string Afile = "A_" + std::string(argv[1]) + "_rounded.csv";
    std::string bfile = "b_" + std::string(argv[1]) + "_rounded.csv";
    std::string startfile = "start_" + std::string(argv[1]) + "_rounded.csv";
    std::string Nfile = "N_" + std::string(argv[1]) + "_rounded.csv";
    std::string p_shiftfile = "p_shift_" + std::string(argv[1]) + "_rounded.csv";
    std::string stringSeed = std::string(argv[2]);


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


//    while (true) {
        markovChain.draw(randomNumberGenerator, 10000, A.cols() * 200);
        markovChain.writeHistory(
                hops::FileWriterFactory::createFileWriter(std::string(argv[1]) + stringSeed, hops::FileWriterType::Csv).get());
        markovChain.clearHistory();
//    }
}
