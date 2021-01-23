#include <any>
#include <boost/program_options.hpp>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileReader/Hdf5Reader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include "hops-sampler.hpp"

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else

#include <experimental/filesystem>
#include <hops/PolytopePreprocessing/MaximumVolumeEllipsoid.hpp>

namespace fs = std::experimental::filesystem;
#endif

std::map<std::string, std::any> parseCommandLineOptions(int argc, char **argv);

void runUniformSampling(std::map<std::string, std::any> arguments, const hops::FileWriter *fileWriter);

/**
 * @Brief Use option --help to see usage.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    auto arguments = parseCommandLineOptions(argc, argv);
    auto fileWriter = std::any_cast<std::shared_ptr<const hops::FileWriter>>(arguments["fileWriter"]);
    runUniformSampling(arguments, fileWriter.get());
}

std::map<std::string, std::any> parseCommandLineOptions(int argc, char **argv) {
    std::map<std::string, std::any> arguments;

    boost::program_options::options_description optionsDescription("options");

    optionsDescription.add_options()
            ("help,h", "Show help message.")
            ("input-path,i",
             boost::program_options::value<std::string>(),
             "Either a path to an hdf5 file containing the polytope (as produced by PolyRound)"
             "or path do directory containing the polytope files in csv format. For an example see the e_coli_core directory in the resources directory.")
            ("output-path,o",
             boost::program_options::value<std::string>(),
             "Path to outputfile with either .hdf5 or .csv ending.")
            ("number-of-samples,n",
             boost::program_options::value<long>(),
             "Number of samples to generate.")
            ("thinning-factor,t",
             boost::program_options::value<long>(),
             "0 for no thinning, else thinning will be the Polytope dimensions times the thinning Factor")
            ("random-seed,r",
             boost::program_options::value<int>(),
             "0 for no thinning, else thinning will be the Polytope dimensions times the thinning Factor")
            ("rounding,r",
             boost::program_options::value<bool>(),
             "Flag, true to apply rounding and false otherwise (defaults to true).");

    boost::program_options::variables_map commandLineOptions;
    boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, optionsDescription),
            commandLineOptions
    );
    boost::program_options::notify(commandLineOptions);

    if (commandLineOptions.count("help")) {
        std::cout << "HOPS uniform sampler for polytopes" << std::endl;
        std::cout << optionsDescription << std::endl;
        exit(1);
    }

    if (commandLineOptions.count("input-path")) {
        auto inputPath = commandLineOptions["input-path"].as<std::string>();
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        Eigen::VectorXd start;
        Eigen::MatrixXd transformation;
        Eigen::VectorXd shift;

        std::string fileExtension = inputPath.substr(inputPath.find_last_of('.') + 1);
        if (fileExtension == "hdf5" || fileExtension == "h5") {
            A = hops::Hdf5Reader::read<decltype(A)>(inputPath, "A");
            b = hops::Hdf5Reader::read<decltype(b)>(inputPath, "b");
            transformation = hops::Hdf5Reader::read<decltype(transformation)>(inputPath, "transformation");
            shift = hops::Hdf5Reader::read<decltype(shift)>(inputPath, "shift");
            try {
                start = hops::Hdf5Reader::read<decltype(start)>(inputPath, "start");
            } catch (...) {
                start = hops::LinearProgramFactory::createLinearProgram(A,
                                                                        b)->calculateChebyshevCenter().optimalParameters;
            }
        } else {
            // Assumes input-path is a directory containing csv-files.
            if (inputPath.back() == fs::path::preferred_separator) {
                inputPath = inputPath.substr(0, inputPath.size() - 1);
            }
            fs::path inputPathImpl(inputPath);
            std::string modelName = inputPathImpl.filename();

            std::string Afile = inputPathImpl / fs::path("A_" + modelName + "_rounded.csv");
            std::string bfile = inputPathImpl / fs::path("b_" + modelName + "_rounded.csv");
            std::string transformFile =
                    inputPathImpl / fs::path("T_" + modelName + "_rounded.csv");
            std::string shiftFile =
                    inputPathImpl / fs::path("shift_" + modelName + "_rounded.csv");
            std::string startFile =
                    inputPathImpl / fs::path("start_" + modelName + "_rounded.csv");

            A = hops::CsvReader::readMatrix<decltype(A)>(Afile);
            b = hops::CsvReader::readVector<decltype(b)>(bfile);
            transformation = hops::CsvReader::readMatrix<decltype(transformation)>(transformFile);
            shift = hops::CsvReader::readVector<decltype(shift)>(startFile);
            try {
                start = hops::CsvReader::readVector<decltype(start)>(startFile);
            } catch (...) {
                start = hops::LinearProgramFactory::createLinearProgram(A, b)->calculateChebyshevCenter()
                        .optimalParameters;
            }
        }
        arguments["A"] = A;
        arguments["b"] = b;
        arguments["transformation"] = transformation;
        arguments["shift"] = shift;
        arguments["start"] = start;
    } else {
        std::cerr << "Missing polytope input, see --help." << std::endl;
        exit(1);
    }

    std::shared_ptr<const hops::FileWriter> fileWriter;
    if (commandLineOptions.count("output-path")) {
        auto outputPath = commandLineOptions["output-path"].as<std::string>();
        std::string fileType = fs::path(outputPath).extension();
        outputPath = fs::path(outputPath).parent_path() / fs::path(outputPath).stem();
        if (fileType == ".csv") {
            fileWriter = hops::FileWriterFactory::createFileWriter(outputPath, hops::FileWriterType::CSV);
        } else if (fileType == ".hdf5") {
            fileWriter = hops::FileWriterFactory::createFileWriter(outputPath, hops::FileWriterType::HDF5);
        } else {
            std::cerr << "Wrong output filetype, see --help." << std::endl;
        }
    } else {
        std::cerr << "Missing output path, see --help." << std::endl;
    }
    if (!fileWriter) {
        std::cerr << "Failed creating writer object." << std::endl;
        exit(1);
    }
    arguments["fileWriter"] = fileWriter;

    if (commandLineOptions.count("number-of-samples")) {
        arguments["numberOfSamples"] = commandLineOptions["number-of-samples"].as<long>();
    } else {
        std::cerr << "number of samples argument required, see --help" << std::endl;
        exit(1);
    }

    if (commandLineOptions.count("thinning-factor")) {
        long thinningFactor = commandLineOptions["thinning-factor"].as<long>();
        arguments["thinning"] = static_cast<long>(thinningFactor *
                                                  std::any_cast<Eigen::MatrixXd>(arguments["A"]).cols());
    } else {
        arguments["thinning"] = static_cast<long>(1);
    }

    if (commandLineOptions.count("random-seed")) {
        arguments["randomSeed"] = commandLineOptions["random-seed"].as<int>();
    } else {
        arguments["randomSeed"] = static_cast<int>(std::random_device()());
    }

    arguments["samplingAlgorithm"] = "CHRR";
    arguments["rounding"] = true;

    // Overwrites default rounding settings if flag is given.
    if (commandLineOptions.count("rounding")) {
        arguments["rounding"] = commandLineOptions["rounding"].as<bool>();
    }

    return arguments;
}

void runUniformSampling(std::map<std::string, std::any> arguments, const hops::FileWriter *fileWriter) {
    // Case: no rounding transformation is included in input and hops should not round
    if (!std::any_cast<bool>(arguments["rounding"])) {
        auto states = sampleUniformly(
                std::any_cast<Eigen::MatrixXd>(arguments["A"]),
                std::any_cast<Eigen::VectorXd>(arguments["b"]),
                std::any_cast<Eigen::VectorXd>(arguments["start"]),
                std::any_cast<long>(arguments["numberOfSamples"]),
                std::any_cast<long>(arguments["thinning"]),
                std::any_cast<int>(arguments["randomSeed"])
        );
        fileWriter->write("states", states);

    }
        // Case: rounding trafo is included in input
    else if (arguments.find("transformation") != arguments.end()) {
        auto states = sampleUniformly(
                std::any_cast<Eigen::MatrixXd>(arguments["A"]),
                std::any_cast<Eigen::VectorXd>(arguments["b"]),
                std::any_cast<Eigen::VectorXd>(arguments["start"]),
                std::any_cast<Eigen::MatrixXd>(arguments["transformation"]),
                std::any_cast<Eigen::VectorXd>(arguments["shift"]),
                std::any_cast<long>(arguments["numberOfSamples"]),
                std::any_cast<long>(arguments["thinning"]),
                std::any_cast<int>(arguments["randomSeed"])
        );
        fileWriter->write("states", states);
    }
        // Case: no rounding trafo is included in input, but hops should approximate rounding
    else if (arguments.find("transformation") == arguments.end()) {
        auto A = std::any_cast<Eigen::MatrixXd>(arguments["A"]);
        auto b = std::any_cast<Eigen::VectorXd>(arguments["b"]);
        auto transformation = hops::MaximumVolumeEllipsoid<double>::construct(A, b, 1e6).getRoundingTransformation();
        // Transforms start into rounded space.
        // Requires rounding transformation (the result of a cholesky decomposition) to be stored as lower diagonal L of LLT and not UUT
        auto start = transformation.template triangularView<Eigen::Lower>().solve(
                std::any_cast<Eigen::VectorXd>(arguments["start"]));

        auto states = sampleUniformly(
                decltype(A)(A * transformation),
                b,
                start,
                transformation,
                Eigen::VectorXd::Zero(A.cols()),
                std::any_cast<long>(arguments["numberOfSamples"]),
                std::any_cast<long>(arguments["thinning"]),
                std::any_cast<int>(arguments["randomSeed"])
        );
        fileWriter->write("states", states);
    }
}
