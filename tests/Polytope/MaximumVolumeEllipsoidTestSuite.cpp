#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MaximumVolumeEllipsoidTestSuite

#include <boost/test/unit_test.hpp>

#include "hops/hops.hpp"
#include "hops/FileReader/CsvReader.hpp"

namespace {
    Eigen::MatrixXd createEcoliMatrix();

    Eigen::VectorXd createEcoliVector();

    Eigen::VectorXd createEcoliStart();

    Eigen::MatrixXd createEcoliMatlabE();

    Eigen::MatrixXd createEcoliMatlabE2();
}

BOOST_AUTO_TEST_SUITE(MaximumVolumeEllipsoid)

    BOOST_AUTO_TEST_CASE(test2DSimplexSameResultAsCobraToolbox) {
        Eigen::MatrixXd expectedRoundingTransformation(2, 2);
        expectedRoundingTransformation << 0.333333302249001, 0.,
                -0.166666635206467, 0.288675116865272;
        Eigen::VectorXd expectedCenter(2);
        expectedCenter << 0.333333335613954, 0.333333335613954;

        Eigen::MatrixXd A(3, 2);
        A << -1, 0,
                0, -1,
                1, 1;
        Eigen::VectorXd b(3);
        b << 0, 0, 1;

        Eigen::VectorXd start(2);
        start << 0.25, 0.25;

        auto maximumVolumeEllipsoid = hops::MaximumVolumeEllipsoid<double>::construct(A, b, 5000, start);

        BOOST_CHECK_LE((maximumVolumeEllipsoid.getRoundingTransformation() - expectedRoundingTransformation).norm(),
                       1e-7);
        BOOST_CHECK_LE((maximumVolumeEllipsoid.getCenter() - expectedCenter).norm(), 1e-7);
    }

    BOOST_AUTO_TEST_CASE(test4DSimplexSameResultAsCobraToolbox) {
        Eigen::MatrixXd expectedRoundingTransformation(4, 4);
        expectedRoundingTransformation << 0.199999997299948, 0, 0, 0,
                -0.049999999324932, 0.193649164696071, 0, 0,
                -0.049999999324932, -0.064549721565262, 0.182574183370304, 0,
                -0.049999999324932, -0.064549721565262, -0.091287091684950, 0.158113880873998;

        Eigen::VectorXd expectedCenter(4);
        expectedCenter << 0.200000000299977, 0.200000000299977, 0.200000000299977, 0.200000000299977;

        Eigen::MatrixXd A(5, 4);
        A << -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1,
                1, 1, 1, 1;
        Eigen::VectorXd b(5);
        b << 0, 0, 0, 0, 1;

        Eigen::VectorXd start(4);
        start << 0.1, 0.1, 0.1, 0.1;

        auto maximumVolumeEllipsoid = hops::MaximumVolumeEllipsoid<double>::construct(A, b, 5000, start);

        BOOST_CHECK_LE((maximumVolumeEllipsoid.getRoundingTransformation() - expectedRoundingTransformation).norm(),
                       1e-7);
        BOOST_CHECK_LE((maximumVolumeEllipsoid.getCenter() - expectedCenter).norm(), 1e-7);
    }

    BOOST_AUTO_TEST_CASE(testAuthorsDemo) {
        Eigen::MatrixXd A(18, 2);
        A << -0.890496033275099, -1.00806441730899, 0.139061858656017, 0.944284824573101, -0.236144297158048,
                -2.42395713384503, -0.0754591290328577, -0.223831428498817, -0.358571912766115, 0.0580698827354712,
                -2.07763485529806, -0.424614015056491, -0.143545710236981, -0.202917945340724, 1.39334147492104,
                -1.51307697899823, 0.651804091657409, -1.12635186101317, -0.377133557739639, -0.815002157728395,
                -0.661443059471046, 0.366614269701525, 0.248957976189754, -0.586106758460856, -0.383516157216677,
                1.53740902604256, -0.528479803889375, 0.140071528525743, 0.0553883642703117, -1.86276666587731,
                1.25376857106666, -0.454193096983248, -2.52000363943994, -0.652074105236213, 0.584856120354184,
                0.103317876922552;

        Eigen::VectorXd b(18);
        b << -0.755972280243298, 1.27585691710246, -0.181010860594784, 0.237445950423737, 0.0217277772435122,
                -1.46201477997428, 0.236818223531106, 1.50419911335932, 0.473911340657419, -0.421851787336940,
                -0.0358193558740663, 0.978031282093556, 0.877954743133157, -0.157160347511024, -0.116894695624955,
                1.85188802037506, -2.00206974955625, 1.15734049563925;

        Eigen::VectorXd start(2);
        start << 0.843869498874359, 0.173900248461784;

        auto maximumVolumeEllipsoid = hops::MaximumVolumeEllipsoid<double>::construct(A, b, 50, start);

        BOOST_CHECK_LE(std::abs(maximumVolumeEllipsoid.computeVolume() - 0.464049344828066),
                       1e-8 * 0.464049344828066);

        Eigen::MatrixXd matlabE2(2, 2);
        matlabE2 << 0.276885814557154, 0.106004009953758, 0.106004009953758, 0.119383276023007;

        Eigen::MatrixXd matlabE(2, 2);
        matlabE << 0.526199405698215, 0, 0.201452165863056, 0.280713912893698;

        const double maxErrorE = (matlabE - maximumVolumeEllipsoid.getRoundingTransformation()).cwiseAbs().maxCoeff();
        const double maxErrorE2 = (matlabE2 - maximumVolumeEllipsoid.getEllipsoid()).cwiseAbs().maxCoeff();
        BOOST_CHECK_LE(maxErrorE, 1e-8);
        BOOST_CHECK_LE(maxErrorE2, 1e-8);
    }


    BOOST_AUTO_TEST_CASE(testEcoli) {
        Eigen::MatrixXd ecoliMatrix = createEcoliMatrix();
        Eigen::VectorXd ecoliVector = createEcoliVector();
        Eigen::VectorXd start = createEcoliStart();
        auto maximumVolumeEllipsoid = hops::MaximumVolumeEllipsoid<double>::construct(
                ecoliMatrix,
                ecoliVector,
                12,
                start
        );

        const double vol = maximumVolumeEllipsoid.computeVolume();
        BOOST_CHECK_LE(std::abs(vol - 1.05772409651363e+28), 1e-10 * 1.05772409651363e+28);

        Eigen::MatrixXd matlabE = createEcoliMatlabE();
        Eigen::MatrixXd matlabE2 = createEcoliMatlabE2();
        const double maxErrorE = (matlabE - maximumVolumeEllipsoid.getRoundingTransformation()).cwiseAbs().maxCoeff();
        const double maxErrorE2 = (matlabE2 - maximumVolumeEllipsoid.getEllipsoid()).cwiseAbs().maxCoeff();
        BOOST_CHECK_LE(maxErrorE, 1e-10);
        BOOST_CHECK_LE(maxErrorE2, 1e-10);
    }

#if HOPS_CLP_FOUND || HOPS_GUROBI_FOUND

    BOOST_AUTO_TEST_CASE(testDifferentStartingPointForEcoli) {
        const int maximumNumberOfIterationsToRun = 30000;
        Eigen::MatrixXd A = createEcoliMatrix();
        Eigen::VectorXd b = createEcoliVector();
        Eigen::VectorXd start = createEcoliStart();

        auto maximumVolumeEllipsoid = hops::MaximumVolumeEllipsoid<double>::construct(A,
                                                                                      b,
                                                                                      maximumNumberOfIterationsToRun,
                                                                                      start,
                                                                                      1e-6);
        const double vol1 = maximumVolumeEllipsoid.computeVolume();

        const Eigen::MatrixXd matlabE = createEcoliMatlabE();
        const Eigen::MatrixXd matlabE2 = createEcoliMatlabE2();

        auto linearProgram = hops::LinearProgramFactory::createLinearProgram(A, b);
        auto linearProgramSolution = linearProgram->solve(start);
        Eigen::VectorXd start2 = (start + linearProgramSolution.optimalParameters) / 2;

        auto maximumVolumeEllipsoid2 = hops::MaximumVolumeEllipsoid<double>::construct(A,
                                                                                       b,
                                                                                       maximumNumberOfIterationsToRun,
                                                                                       start2,
                                                                                       1e-6);

        const double vol2 = maximumVolumeEllipsoid2.computeVolume();
        const double maxCenterError = (maximumVolumeEllipsoid.getCenter() -
                                       maximumVolumeEllipsoid2.getCenter()).lpNorm<Eigen::Infinity>();

        BOOST_CHECK(maximumVolumeEllipsoid.hasConverged());
        BOOST_CHECK(maximumVolumeEllipsoid2.hasConverged());
        BOOST_CHECK_LE(std::abs(vol2 - vol1) / vol2, 0.01);
        BOOST_CHECK_LE(maxCenterError, 0.01);
        BOOST_CHECK_LE((maximumVolumeEllipsoid.getRoundingTransformation() -
                        maximumVolumeEllipsoid2.getRoundingTransformation())
                               .lpNorm<Eigen::Infinity>(), 0.05);
    }

#endif //HOPS_CLP_FOUND || HOPS_GUROBI_FOUND

BOOST_AUTO_TEST_SUITE_END()

namespace {
    Eigen::MatrixXd createEcoliMatrix() {
        return hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../../resources/expectedMVEs/EcoliMatrix.csv");
    }

    Eigen::VectorXd createEcoliVector() {
        Eigen::VectorXd b(81);
        b << 0.436742000000000, -4.88717000000000, -0.0722710000000000, 0.257198000000000, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.185230000000000, 0.0548100000000000,
                100.0 * Eigen::VectorXd::Ones(27 * 2);
        return b;
    }

    Eigen::VectorXd createEcoliStart() {
        Eigen::VectorXd start(27);
        start << 81.1313199895542, 0.0367081159679601, 81.1313199895542, 0.0537389137324980, 81.1313199895542,
                0.0923061170055630, 0.0274050000000041, 81.1313199895542, 0.0717698774221035, 0, 0, 81.1313199895542,
                81.1313199895542, 81.1313199895542, 81.1313199895542, -20.5265158112708, 0, 20.5265158112708,
                81.1313199895542, 81.1313199895542, 81.1313199895542, 81.1313199895542, 81.1313199895542,
                20.5265158112708, 81.1313199895542, 81.1313199895542, 50.7152556798561;
        return start;
    }

    Eigen::MatrixXd createEcoliMatlabE() {
        return hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../../resources/expectedMVEs/matlabE.csv");
    }

    Eigen::MatrixXd createEcoliMatlabE2() {
        return hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../../resources/expectedMVEs/matlabE2.csv");
    }
}
