#define BOOST_TEST_MODULE IsCalculateExpectedFisherInformationAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/Model/IsCalculateLogExpectedFisherInformationAvailable.hpp>

BOOST_AUTO_TEST_SUITE(IsCalculateExpectedFisherInformationAvailable)

    BOOST_AUTO_TEST_CASE(WhenCalculateExpectedFisherInformationIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(hops::IsCalculateExpectedFisherInformationAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenCalculateExpectedFisherInformationHasWrongSignature) {
        class Mock {
        public:
            double computeExpectedFisherInformation(Mock);
        };

        BOOST_CHECK(hops::IsCalculateExpectedFisherInformationAvailable<Mock>::value == false);
    }

    BOOST_AUTO_TEST_CASE(WhenCalculateExpectedFisherInformationIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            using VectorType = double;
            using MatrixType = int;

            MatrixType computeExpectedFisherInformation(const VectorType &);
        };

        BOOST_CHECK(hops::IsCalculateExpectedFisherInformationAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
