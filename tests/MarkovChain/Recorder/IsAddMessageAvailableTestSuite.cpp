#define BOOST_TEST_MODULE IsAddMessageAvailableTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/Recorder/IsAddMessageAvailabe.hpp>

BOOST_AUTO_TEST_SUITE(IsAddMessageAvailableTestSuite)

    BOOST_AUTO_TEST_CASE(WhenAddMessageIsNotAvailable) {
        class Mock {
        public:
        };
        BOOST_CHECK(!hops::IsAddMessageAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenAddMessageHasWrongSignature) {
        class Mock {
        public:
            void addMessage();
        };

        BOOST_CHECK(!hops::IsAddMessageAvailable<Mock>::value);
    }

    BOOST_AUTO_TEST_CASE(WhenAddMessageIsAvailableWithCorrectTypedef) {
        class Mock {
        public:
            void addMessage(const std::string &);
        };

        BOOST_CHECK(hops::IsAddMessageAvailable<Mock>::value);
    }

BOOST_AUTO_TEST_SUITE_END()
