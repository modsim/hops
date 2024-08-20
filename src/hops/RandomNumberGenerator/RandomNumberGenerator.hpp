#ifndef HOPS_RANDOMNUMBERGENERATOR_HPP
#define HOPS_RANDOMNUMBERGENERATOR_HPP

#include "hops/extern/pcg-cpp/pcg_random.hpp"
#include <array>

namespace hops {
    namespace {
        std::array<unsigned char, 16> stateToBytes(pcg64::state_type state) {
            std::array<unsigned char, 16> bytes;
            std::memcpy(bytes.data(), &state, 16);
            return bytes;
        }

        pcg64::state_type bytesToState(const std::array<unsigned char, 16> &bytes) {
            pcg64::state_type state = 0;
            std::memcpy(&state, bytes.data(), 16);
            return state;
        }
    }// namespace

    struct RandomNumberGenerator {
        typedef pcg64::result_type result_type;
        typedef pcg64::state_type state_type;
        state_type seed_;
        state_type stream_;
        pcg64 rng_;

        explicit RandomNumberGenerator(state_type seed = 0, state_type stream = pcg64(0).stream()) : seed_(seed),
                                                                                                     stream_(stream) {
            rng_ = pcg64(seed_, stream_);
        }

        static constexpr result_type min() {
            return pcg64::min();
        }

        static constexpr result_type max() {
            return pcg64::max();
        }

        [[nodiscard]] state_type getSeed() const {
            return seed_;
        }

        [[nodiscard]] state_type getStream() const {
            return stream_;
        }

        [[nodiscard]] state_type getState() const {
            return rng_ - hops::RandomNumberGenerator(seed_, stream_).rng_;
        }

        [[nodiscard]] std::array<unsigned char, 16> getStateInBytes() const {
            return stateToBytes(rng_ - hops::RandomNumberGenerator(seed_, stream_).rng_);
        }

        [[nodiscard]] std::array<unsigned char, 16> getStreamInBytes() const {
            return  stateToBytes(stream_);
        }


        void setSeed(state_type seed) {
            RandomNumberGenerator::seed_ = seed;
            rng_.seed(this->seed_);
        }

        void seed(state_type seed) {
            RandomNumberGenerator::seed_ = seed;
            rng_.seed(this->seed_);
        }

        void setStream(state_type stream) {
            RandomNumberGenerator::stream_ = stream;
            rng_.set_stream(this->stream_);
        }

        void setStream(const std::array<unsigned char, 16> &bytes) {
            auto stream = bytesToState(bytes);
            RandomNumberGenerator::stream_ = stream;
            rng_.set_stream(this->stream_);
        }

        void setState(const std::array<unsigned char, 16> &bytes) {
            rng_.advance(bytesToState(bytes));
        }

        result_type operator()() {
            return rng_();
        }

        result_type operator-(const RandomNumberGenerator &other) const {
            return this->rng_ - other.rng_;
        }
    };
}// namespace hops

#endif//HOPS_RANDOMNUMBERGENERATOR_HPP
