#define BOOST_TEST_MODULE RandomNumberGeneratorTestSuite
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <fstream>

#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"

BOOST_AUTO_TEST_SUITE(RandomNumberGenerator)

BOOST_AUTO_TEST_CASE(ByteConversionTest) {
    int seed = 5;
    int stream = 1337;
    hops::RandomNumberGenerator rng(seed, stream);
    auto expectedState = rng.getState();
    auto expectedStream = rng.getStream();

    auto actualState = rng.bytesToState(rng.getStateInBytes());
    auto actualStream = rng.bytesToState(rng.getStreamInBytes());

    BOOST_CHECK(actualState == expectedState);
    BOOST_CHECK(actualStream == expectedStream);
}

BOOST_AUTO_TEST_CASE(TestSerialization) {
        int seed = 5;
        int stream = 1337;

        hops::RandomNumberGenerator rng(seed, stream);
        for(int i=0; i<10; ++i) {
            rng();
        }

        std::fstream out;
        out.open("rng_test.txt", std::ios::out);
        rng.serialize(out);
        out.close();

        hops::RandomNumberGenerator loaded_rng(seed+1, stream+1);
        BOOST_CHECK(loaded_rng.getState() != rng.getState());
        BOOST_CHECK(loaded_rng.getStream() != rng.getStream());

        std::fstream in("rng_test.txt", std::ios::in);
        loaded_rng = hops::RandomNumberGenerator::deserialize(in);
        in.close();

        BOOST_CHECK(loaded_rng.getState() == rng.getState());
        BOOST_CHECK(loaded_rng.getStream() == rng.getStream());

        for(size_t i=0; i<100; ++i) {
            rng();
            loaded_rng();
        }

        BOOST_CHECK(loaded_rng.getState() == rng.getState());
        BOOST_CHECK(loaded_rng.getStream() == rng.getStream());
}

BOOST_AUTO_TEST_SUITE_END()
