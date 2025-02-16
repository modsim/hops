#ifndef HOPS_RANDOMNUMBERGENERATOR_HPP
#define HOPS_RANDOMNUMBERGENERATOR_HPP

#include "hops/extern/pcg-cpp/pcg_random.hpp"
#include <array>

namespace hops {
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

        static constexpr result_type

        min() {
            return pcg64::min();
        }

        static constexpr result_type

        max() {
            return pcg64::max();
        }

        void serialize(std::ostream &out) const {
            auto seed_bytes = stateToBytes(this->seed_);
            auto stream_bytes = stateToBytes(this->stream_);
            auto state_bytes = this->getStateInBytes();
            out.write(seed_bytes.data(), seed_bytes.size());
            out.write(stream_bytes.data(), stream_bytes.size());
            out.write(state_bytes.data(), state_bytes.size());
        }

        static RandomNumberGenerator deserialize(std::istream &in) {
            std::array<char, 16> seed_bytes;
            std::array<char, 16> stream_bytes;
            std::array<char, 16> state_bytes;
            in.read(seed_bytes.data(), seed_bytes.size());
            in.read(stream_bytes.data(), stream_bytes.size());
            in.read(state_bytes.data(), state_bytes.size());
            auto seed = bytesToState(seed_bytes);
            auto stream = bytesToState(stream_bytes);
            auto state = bytesToState(state_bytes);
            RandomNumberGenerator rng(seed,stream);
            rng.setState(state);
            return rng;
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

        [[nodiscard]] std::array<char, 16> getStateInBytes() const {
            return stateToBytes(rng_ - hops::RandomNumberGenerator(seed_, stream_).rng_);
        }

        [[nodiscard]] std::array<char, 16> getStreamInBytes() const {
            return stateToBytes(stream_);
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

        void setStream(const std::array<char, 16> &bytes) {
            auto stream = bytesToState(bytes);
            RandomNumberGenerator::stream_ = stream;
            rng_.set_stream(this->stream_);
        }

        void setState(const std::array<char, 16> &bytes) {
            rng_.advance(bytesToState(bytes));
        }

        void setState(const state_type state) {
            rng_.advance(state);
        }

        result_type operator()() {
            return rng_();
        }

        result_type operator-(const RandomNumberGenerator &other) const {
            return static_cast<result_type>(this->rng_ - other.rng_);
        }

        static std::array<char, 16> stateToBytes(state_type state) {
            std::array<char, 16> bytes;
            std::memcpy(bytes.data(), &state, 16);
            return bytes;
        }

        static state_type bytesToState(const std::array<char, 16> &bytes) {
            state_type state = 0;
            std::memcpy(&state, bytes.data(), 16);
            return state;
        }

        static std::string stringRepresentation(state_type value) {
            if (value == static_cast<state_type>(0)) {
                return "0";
            }
            std::string representation;
	        auto short_value = static_cast<long>(value);
            while (short_value > static_cast<decltype(short_value)>(0)) {
                representation.insert(representation.begin(), '0' + (short_value % static_cast<decltype(short_value)>(10)));
                short_value = short_value/static_cast<state_type>(10);
            }
            return representation;
        }

    };
}// namespace hops

#endif//HOPS_RANDOMNUMBERGENERATOR_HPP
