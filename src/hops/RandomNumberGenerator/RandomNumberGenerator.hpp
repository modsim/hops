#ifndef HOPS_RANDOMNUMBERGENERATOR_HPP
#define HOPS_RANDOMNUMBERGENERATOR_HPP

#include "hops/extern/pcg-cpp/pcg_random.hpp"

#include <array>
#include <cstring>
#include <ostream>
#include <istream>
#include <string>
#include <algorithm>

namespace hops {
    struct RandomNumberGenerator {
        using result_type = pcg64::result_type;
        using state_type = pcg64::state_type;
        using state_bytes_type = std::array<char, sizeof(state_type)>;

        state_type seed_;
        state_type stream_;
        pcg64 rng_;

        explicit RandomNumberGenerator(
            state_type seed = state_type(0),
            state_type stream = pcg64(0).stream())
            : seed_(seed), stream_(stream), rng_(seed_, stream_) {}

        static constexpr result_type min() {
            return pcg64::min();
        }

        static constexpr result_type max() {
            return pcg64::max();
        }

        void serialize(std::ostream& out) const {
            const auto seed_bytes = stateToBytes(seed_);
            const auto stream_bytes = stateToBytes(stream_);
            const auto state_bytes = getStateInBytes();

            out.write(seed_bytes.data(), static_cast<std::streamsize>(seed_bytes.size()));
            out.write(stream_bytes.data(), static_cast<std::streamsize>(stream_bytes.size()));
            out.write(state_bytes.data(), static_cast<std::streamsize>(state_bytes.size()));
        }

        static RandomNumberGenerator deserialize(std::istream& in) {
            state_bytes_type seed_bytes{};
            state_bytes_type stream_bytes{};
            state_bytes_type state_bytes{};

            in.read(seed_bytes.data(), static_cast<std::streamsize>(seed_bytes.size()));
            in.read(stream_bytes.data(), static_cast<std::streamsize>(stream_bytes.size()));
            in.read(state_bytes.data(), static_cast<std::streamsize>(state_bytes.size()));

            const auto seed = bytesToState(seed_bytes);
            const auto stream = bytesToState(stream_bytes);
            const auto state = bytesToState(state_bytes);

            RandomNumberGenerator rng(seed, stream);
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

        [[nodiscard]] state_bytes_type getStateInBytes() const {
            return stateToBytes(getState());
        }

        [[nodiscard]] state_bytes_type getStreamInBytes() const {
            return stateToBytes(stream_);
        }

        void setSeed(state_type seed) {
            seed_ = seed;
            rng_ = pcg64(seed_, stream_);
        }

        void seed(state_type seed) {
            setSeed(seed);
        }

        void setStream(state_type stream) {
            stream_ = stream;
            rng_ = pcg64(seed_, stream_);
        }

        void setStream(const state_bytes_type& bytes) {
            setStream(bytesToState(bytes));
        }

        void setState(const state_bytes_type& bytes) {
            rng_.advance(bytesToState(bytes));
        }

        void setState(state_type state) {
            rng_.advance(state);
        }

        result_type operator()() {
            return rng_();
        }

        state_type operator-(const RandomNumberGenerator& other) const {
            return rng_ - other.rng_;
        }

        static state_bytes_type stateToBytes(const state_type& state) {
            state_bytes_type bytes{};
            std::memcpy(bytes.data(), &state, sizeof(state));
            return bytes;
        }

        static state_type bytesToState(const state_bytes_type& bytes) {
            state_type state{};
            std::memcpy(&state, bytes.data(), sizeof(state));
            return state;
        }

        static std::string stringRepresentation(state_type value) {
            if (value == state_type(0)) {
                return "0";
            }

            std::string representation;
            const state_type ten = static_cast<state_type>(10);

            while (value > state_type(0)) {
                const auto digit = static_cast<unsigned>(value % ten);
                representation.push_back(static_cast<char>('0' + digit));
                value /= ten;
            }

            std::reverse(representation.begin(), representation.end());
            return representation;
        }
    };
} // namespace hops

#endif // HOPS_RANDOMNUMBERGENERATOR_HPP