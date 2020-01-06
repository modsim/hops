#ifndef NUPS_FIXTURES_HPP
#define NUPS_FIXTURES_HPP

#include <celero/Celero.h>
#include <nups/FileReader/CsvReader.hpp>
#include <nups/MarkovChain/MarkovChainFactory.hpp>
#include <nups/PolytopeSpace/PolytopeSpace.hpp>
#include <nups/Transformation/Transformation.hpp>


template<typename ModelFiles, typename Matrix, typename Vector>
class PolytopeSpaceFixture {
public:
    PolytopeSpaceFixture() {
        polytopeSpace.A = nups::CsvReader::readMatrix<Matrix>(std::string(ModelFiles::A));
        polytopeSpace.b = nups::CsvReader::readVector<Vector>(std::string(ModelFiles::b));
        polytopeSpace.startingPoint = nups::CsvReader::readVector<Vector>(std::string(ModelFiles::startingPoint));
//        polytopeSpace.N = nups::CsvReader::readMatrix<Matrix>(std::string(ModelFiles::N));
//        polytopeSpace.shift = nups::CsvReader::readVector<Vector>(std::string(ModelFiles::p_shift));

        polytopeSpace.roundedA = nups::CsvReader::readMatrix<Matrix>(std::string(ModelFiles::roundedA));
        polytopeSpace.roundedb = nups::CsvReader::readVector<Vector>(std::string(ModelFiles::roundedb));
        polytopeSpace.roundedStartingPoint = nups::CsvReader::readVector<Vector>(std::string(ModelFiles::roundedStartingPoint));
        polytopeSpace.roundedT = nups::CsvReader::readMatrix<Matrix>(std::string(ModelFiles::roundedT));
        polytopeSpace.roundedN = nups::CsvReader::readMatrix<Matrix>(std::string(ModelFiles::roundedN));
        polytopeSpace.roundedShift = nups::CsvReader::readVector<Vector>(std::string(ModelFiles::roundedp_shift));
    }

    nups::PolytopeSpace<Matrix, Vector> polytopeSpace;
};

template<typename PolytopeSpaceFixture, nups::MarkovChainType markovChainType>
class MarkovChainFixture : public celero::TestFixture {
public:
    MarkovChainFixture() {
        markovChain = nups::MarkovChainFactory::createMarkovChain(
                polytopeSpaceFixture.polytopeSpace,
                markovChainType
        );
    }

    PolytopeSpaceFixture polytopeSpaceFixture;
    std::unique_ptr<nups::MarkovChain> markovChain;
    nups::RandomNumberGenerator randomNumberGenerator{std::random_device{}()};
};

struct Recon3D {
    static constexpr const char *name = "Recon3D_301";

    static constexpr const char *A = "../resources/Recon3D_301/A_Recon3D_301_unrounded.csv";
    static constexpr const char *b = "../resources/Recon3D_301/b_Recon3D_301_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/Recon3D_301/start_Recon3D_301_unrounded.csv";
    static constexpr const char *N = "../resources/Recon3D_301/N_Recon3D_301_unrounded.csv";
    static constexpr const char *p_shift = "../resources/Recon3D_301/p_shift_Recon3D_301_unrounded.csv";

    static constexpr const char *roundedA = "../resources/Recon3D_301/A_Recon3D_301_rounded.csv";
    static constexpr const char *roundedb = "../resources/Recon3D_301/b_Recon3D_301_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/Recon3D_301/start_Recon3D_301_rounded.csv";
    static constexpr const char *roundedT = "../resources/Recon3D_301/T_Recon3D_301_rounded.csv";
    static constexpr const char *roundedN = "../resources/Recon3D_301/N_Recon3D_301_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/Recon3D_301/p_shift_Recon3D_301_rounded.csv";
};

struct Recon2 {
    static constexpr const char *name = "Recon2.v04";
    static constexpr const char *A = "../resources/Recon2.v04/A_Recon2.v04_unrounded.csv";
    static constexpr const char *b = "../resources/Recon2.v04/b_Recon2.v04_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/Recon2.v04/start_Recon2.v04_unrounded.csv";
    static constexpr const char *roundedA = "../resources/Recon2.v04/A_Recon2.v04_rounded.csv";
    static constexpr const char *roundedb = "../resources/Recon2.v04/b_Recon2.v04_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/Recon2.v04/start_Recon2.v04_rounded.csv";
    static constexpr const char *roundedT = "../resources/Recon2.v04/T_Recon2.v04_rounded.csv";
    static constexpr const char *roundedN = "../resources/Recon2.v04/N_Recon2.v04_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/Recon2.v04/p_shift_Recon2.v04_rounded.csv";

    // Wrong files, since they are not needed right now
    static constexpr const char *N = "../resources/Recon2.v04/N_Recon2.v04_rounded.csv";
    static constexpr const char *p_shift = "../resources/Recon2.v04/p_shift_Recon2.v04_rounded.csv";
};

struct RECON1 {
    static constexpr const char *name = "RECON1";
    static constexpr const char *A = "../resources/RECON1/A_RECON1_unrounded.csv";
    static constexpr const char *b = "../resources/RECON1/b_RECON1_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/RECON1/start_RECON1_unrounded.csv";
    static constexpr const char *N = "../resources/RECON1/N_RECON1_unrounded.csv";
    static constexpr const char *p_shift = "../resources/RECON1/p_shift_RECON1_unrounded.csv";
    static constexpr const char *roundedA = "../resources/RECON1/A_RECON1_rounded.csv";
    static constexpr const char *roundedb = "../resources/RECON1/b_RECON1_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/RECON1/start_RECON1_rounded.csv";
    static constexpr const char *roundedT = "../resources/RECON1/T_RECON1_rounded.csv";
    static constexpr const char *roundedN = "../resources/RECON1/N_RECON1_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/RECON1/p_shift_RECON1_rounded.csv";
};

struct iJO1366 {
    static constexpr const char *name = "iJO1366";
    static constexpr const char *A = "../resources/iJO1366/A_iJO1366_unrounded.csv";
    static constexpr const char *b = "../resources/iJO1366/b_iJO1366_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/iJO1366/start_iJO1366_unrounded.csv";
    static constexpr const char *N = "../resources/iJO1366/N_iJO1366_unrounded.csv";
    static constexpr const char *p_shift = "../resources/iJO1366/p_shift_iJO1366_unrounded.csv";
    static constexpr const char *roundedA = "../resources/iJO1366/A_iJO1366_rounded.csv";
    static constexpr const char *roundedb = "../resources/iJO1366/b_iJO1366_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/iJO1366/start_iJO1366_rounded.csv";
    static constexpr const char *roundedT = "../resources/iJO1366/T_iJO1366_rounded.csv";
    static constexpr const char *roundedN = "../resources/iJO1366/N_iJO1366_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/iJO1366/p_shift_iJO1366_rounded.csv";
};

struct iAT_PLT_636 {
    static constexpr const char *name = "iAT_PLT_636";
    static constexpr const char *A = "../resources/iAT_PLT_636/A_iAT_PLT_636_unrounded.csv";
    static constexpr const char *b = "../resources/iAT_PLT_636/b_iAT_PLT_636_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/iAT_PLT_636/start_iAT_PLT_636_unrounded.csv";
    static constexpr const char *N = "../resources/iAT_PLT_636/N_iAT_PLT_636_unrounded.csv";
    static constexpr const char *p_shift = "../resources/iAT_PLT_636/p_shift_iAT_PLT_636_unrounded.csv";
    static constexpr const char *roundedA = "../resources/iAT_PLT_636/A_iAT_PLT_636_rounded.csv";
    static constexpr const char *roundedb = "../resources/iAT_PLT_636/b_iAT_PLT_636_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/iAT_PLT_636/start_iAT_PLT_636_rounded.csv";
    static constexpr const char *roundedT = "../resources/iAT_PLT_636/T_iAT_PLT_636_rounded.csv";
    static constexpr const char *roundedN = "../resources/iAT_PLT_636/N_iAT_PLT_636_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/iAT_PLT_636/p_shift_iAT_PLT_636_rounded.csv";
};

struct e_coli_core {
    static constexpr const char *name = "e_coli_core";
    static constexpr const char *A = "../resources/e_coli_core/A_e_coli_core_unrounded.csv";
    static constexpr const char *b = "../resources/e_coli_core/b_e_coli_core_unrounded.csv";
    static constexpr const char *roundedA = "../resources/e_coli_core/A_e_coli_core_rounded.csv";
    static constexpr const char *roundedb = "../resources/e_coli_core/b_e_coli_core_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/e_coli_core/start_e_coli_core_rounded.csv";
    static constexpr const char *startingPoint = "../resources/e_coli_core/start_e_coli_core_unrounded.csv";
    static constexpr const char *roundedT = "../resources/e_coli_core/T_e_coli_core_rounded.csv";
    static constexpr const char *roundedN = "../resources/e_coli_core/N_e_coli_core_rounded.csv";
    static constexpr const char *N = "../resources/e_coli_core/N_e_coli_core_unrounded.csv";
    static constexpr const char *roundedp_shift = "../resources/e_coli_core/p_shift_e_coli_core_rounded.csv";
    static constexpr const char *p_shift = "../resources/e_coli_core/p_shift_e_coli_core_unrounded.csv";
};

struct simplex_64D {
    static constexpr const char *name = "simplex_64D";
    static constexpr const char *A = "../resources/simplex_64D/A_simplex_64D_unrounded.csv";
    static constexpr const char *b = "../resources/simplex_64D/b_simplex_64D_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/simplex_64D/start_simplex_64D_unrounded.csv";
    static constexpr const char *T = "../resources/simplex_64D/T_simplex_64D_rounded.csv";
    static constexpr const char *N = "../resources/simplex_64D/N_simplex_64D_rounded.csv";
    static constexpr const char *p_shift = "../resources/simplex_64D/p_shift_simplex_64D_rounded.csv";
    static constexpr const char *roundedA = "../resources/simplex_64D/A_simplex_64D_rounded.csv";
    static constexpr const char *roundedb = "../resources/simplex_64D/b_simplex_64D_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/simplex_64D/start_simplex_64D_rounded.csv";
    static constexpr const char *roundedT = "../resources/simplex_64D/T_simplex_64D_rounded.csv";
    static constexpr const char *roundedN = "../resources/simplex_64D/N_simplex_64D_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/simplex_64D/p_shift_simplex_64D_rounded.csv";
};

struct simplex_256D {
    static constexpr const char *name = "simplex_256D";
    static constexpr const char *A = "../resources/simplex_256D/A_simplex_256D_unrounded.csv";
    static constexpr const char *b = "../resources/simplex_256D/b_simplex_256D_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/simplex_256D/start_simplex_256D_unrounded.csv";
    static constexpr const char *T = "../resources/simplex_256D/T_simplex_256D_rounded.csv";
    static constexpr const char *N = "../resources/simplex_256D/N_simplex_256D_rounded.csv";
    static constexpr const char *p_shift = "../resources/simplex_256D/p_shift_simplex_256D_rounded.csv";
    static constexpr const char *roundedA = "../resources/simplex_256D/A_simplex_256D_rounded.csv";
    static constexpr const char *roundedb = "../resources/simplex_256D/b_simplex_256D_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/simplex_256D/start_simplex_256D_rounded.csv";
    static constexpr const char *roundedT = "../resources/simplex_256D/T_simplex_256D_rounded.csv";
    static constexpr const char *roundedN = "../resources/simplex_256D/N_simplex_256D_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/simplex_256D/p_shift_simplex_256D_rounded.csv";
};

struct simplex_512D {
    static constexpr const char *name = "simplex_512D";
    static constexpr const char *A = "../resources/simplex_512D/A_simplex_512D_unrounded.csv";
    static constexpr const char *b = "../resources/simplex_512D/b_simplex_512D_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/simplex_512D/start_simplex_512D_unrounded.csv";
    static constexpr const char *T = "../resources/simplex_512D/T_simplex_512D_rounded.csv";
    static constexpr const char *N = "../resources/simplex_512D/N_simplex_512D_rounded.csv";
    static constexpr const char *p_shift = "../resources/simplex_512D/p_shift_simplex_512D_rounded.csv";
    static constexpr const char *roundedA = "../resources/simplex_512D/A_simplex_512D_rounded.csv";
    static constexpr const char *roundedb = "../resources/simplex_512D/b_simplex_512D_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/simplex_512D/start_simplex_512D_rounded.csv";
    static constexpr const char *roundedT = "../resources/simplex_512D/T_simplex_512D_rounded.csv";
    static constexpr const char *roundedN = "../resources/simplex_512D/N_simplex_512D_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/simplex_512D/p_shift_simplex_512D_rounded.csv";
};

struct simplex_1024D {
    static constexpr const char *name = "simplex_1024D";
    static constexpr const char *A = "../resources/simplex_1024D/A_simplex_1024D_unrounded.csv";
    static constexpr const char *b = "../resources/simplex_1024D/b_simplex_1024D_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/simplex_1024D/start_simplex_1024D_unrounded.csv";
    static constexpr const char *T = "../resources/simplex_1024D/T_simplex_1024D_rounded.csv";
    static constexpr const char *N = "../resources/simplex_1024D/N_simplex_1024D_rounded.csv";
    static constexpr const char *p_shift = "../resources/simplex_1024D/p_shift_simplex_1024D_rounded.csv";
    static constexpr const char *roundedA = "../resources/simplex_1024D/A_simplex_1024D_rounded.csv";
    static constexpr const char *roundedb = "../resources/simplex_1024D/b_simplex_1024D_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/simplex_1024D/start_simplex_1024D_rounded.csv";
    static constexpr const char *roundedT = "../resources/simplex_1024D/T_simplex_1024D_rounded.csv";
    static constexpr const char *roundedN = "../resources/simplex_1024D/N_simplex_1024D_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/simplex_1024D/p_shift_simplex_1024D_rounded.csv";
};

struct simplex_2048D {
    static constexpr const char *name = "simplex_2048D";
    static constexpr const char *A = "../resources/simplex_2048D/A_simplex_2048D_unrounded.csv";
    static constexpr const char *b = "../resources/simplex_2048D/b_simplex_2048D_unrounded.csv";
    static constexpr const char *startingPoint = "../resources/simplex_2048D/start_simplex_2048D_unrounded.csv";
    static constexpr const char *T = "../resources/simplex_2048D/T_simplex_2048D_rounded.csv";
    static constexpr const char *N = "../resources/simplex_2048D/N_simplex_2048D_rounded.csv";
    static constexpr const char *p_shift = "../resources/simplex_2048D/p_shift_simplex_2048D_rounded.csv";
    static constexpr const char *roundedA = "../resources/simplex_2048D/A_simplex_2048D_rounded.csv";
    static constexpr const char *roundedb = "../resources/simplex_2048D/b_simplex_2048D_rounded.csv";
    static constexpr const char *roundedStartingPoint = "../resources/simplex_2048D/start_simplex_2048D_rounded.csv";
    static constexpr const char *roundedT = "../resources/simplex_2048D/T_simplex_2048D_rounded.csv";
    static constexpr const char *roundedN = "../resources/simplex_2048D/N_simplex_2048D_rounded.csv";
    static constexpr const char *roundedp_shift = "../resources/simplex_2048D/p_shift_simplex_2048D_rounded.csv";
};

#endif //NUPS_FIXTURES_HPP
