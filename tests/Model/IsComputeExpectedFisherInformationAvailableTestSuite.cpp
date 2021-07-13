#define BOOST_TEST_MODULE IsComputeExpectedFisherInformationAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/Model/IsComputeLogExpectedFisherInformationAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsComputeExpectedFisherInformationAvailable)

    BOOST_AUTO_TEST_CASE(WhenComputeExpectedFisherInformationIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsComputeExpectedFisherInformationAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenComputeExpectedFisherInformationHasWrongSignature) {
        class Mock {
        public:
            double computeExpectedFisherInformation(Mock);
        };

        BOOST_CHECK(hops::IsComputeExpectedFisherInformationAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenComputeExpectedFisherInformationIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            using VectorType = double;
            using MatrixType = int;

            MatrixType computeExpectedFisherInformation(const VectorType &);
        };

        BOOST_CHECK(hops::IsComputeExpectedFisherInformationAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
