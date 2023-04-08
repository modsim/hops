#include <any>
#include <boost/program_options.hpp>
#include "hops-sampler.hpp"

#include <filesystem>
namespace fs = std::filesystem;

//#ifdef __cpp_lib_filesystem
//#include <filesystem>
//namespace fs = std::filesystem;
//#else
//#include <experimental/filesystem>
//namespace fs = std::experimental::filesystem;
//#endif

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
             "Or path do directory containing the polytope files in csv format. For an example see the e_coli_core directory in the resources directory.")
            ("output-path,o",
             boost::program_options::value<std::string>(),
             "Path to outputfile with either .hdf5 or .csv ending.")
            ("number-of-samples,n",
             boost::program_options::value<long>(),
             "Number of samples to generate.")
            ("thinning-factor,t",
             boost::program_options::value<long>(),
             "0 for no thinning, else thinning will be the Polytope dimensions times the thinning Factor")
            ("random-seed,s",
             boost::program_options::value<int>(),
             "seed for the random number generator. If not supplied it will be generated from std::random_device()")
            ("rounding,r",
             boost::program_options::value<bool>(),
             "Flag, true to apply rounding and false otherwise (defaults to true).")
            ("batch-size,b",
             boost::program_options::value<long>(),
             "how many samples to do before storing");

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
        shift = hops::CsvReader::readVector<decltype(shift)>(shiftFile);
        try {
            start = hops::CsvReader::readVector<decltype(start)>(startFile);
        } catch (...) {
            start = hops::LinearProgramFactory::createLinearProgram(A, b)->computeChebyshevCenter()
                    .optimalParameters;
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

    if (commandLineOptions.count("batch-size")) {
        arguments["batch-size"] = commandLineOptions["batch-size"].as<long>();
    } else {
        arguments["batch-size"] = commandLineOptions["number-of-samples"].as<long>();
    }

    return arguments;
}

void runUniformSampling(std::map<std::string, std::any> arguments, const hops::FileWriter *fileWriter) {
    // Case: no rounding transformation is included in input and hops should not round
    if (!std::any_cast<bool>(arguments["rounding"])) {
        auto[randomNumberGenerator, markovChain] = setUpSampling(
                std::any_cast<Eigen::MatrixXd>(arguments["A"]),
                std::any_cast<Eigen::VectorXd>(arguments["b"]),
                std::any_cast<Eigen::VectorXd>(arguments["start"]),
                std::any_cast<int>(arguments["randomSeed"])
        );

        auto[states, times] = sampleUniformBatch(
                std::any_cast<long>(arguments["numberOfSamples"]),
                std::any_cast<long>(arguments["thinning"]),
                randomNumberGenerator,
                markovChain.get()

        );
        fileWriter->write("states", states);
        fileWriter->write("times", times);

    }
        // Case: rounding trafo is included in input
    else if (arguments.find("transformation") != arguments.end()) {
        auto[randomNumberGenerator, markovChain, transformation] = setUpSampling(
                std::any_cast<Eigen::MatrixXd>(arguments["A"]),
                std::any_cast<Eigen::VectorXd>(arguments["b"]),
                std::any_cast<Eigen::VectorXd>(arguments["start"]),
                std::any_cast<Eigen::MatrixXd>(arguments["transformation"]),
                std::any_cast<Eigen::VectorXd>(arguments["shift"]),
                std::any_cast<int>(arguments["randomSeed"])
        );

        long numberOfSamples = std::any_cast<long>(arguments["numberOfSamples"]);
        long batchSize = std::any_cast<long>(arguments["batch-size"]);

        while (numberOfSamples > 0) {
            auto[states, sample_times, transform_times] = sampleUniformBatch(
                    batchSize,
                    std::any_cast<long>(arguments["thinning"]),
                    randomNumberGenerator,
                    markovChain.get(),
                    transformation.get());

            fileWriter->write("states", states);
            fileWriter->write("sample_times", sample_times);
            fileWriter->write("transform_times", transform_times);
            numberOfSamples = numberOfSamples - batchSize;
            if (batchSize > numberOfSamples) {
                batchSize = numberOfSamples;
            }
        }
    }
        // Case: no rounding trafo is included in input, but hops should approximate rounding
    else if (arguments.find("transformation") == arguments.end()) {
        throw std::runtime_error("use hopsy");
    }
}
