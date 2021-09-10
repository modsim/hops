#ifndef HOPS_EXCEPTIONS_HPP
#define HOPS_EXCEPTIONS_HPP

#include <exception>
#include <string>

namespace hops {
    class Exception : public std::exception {
    public:
        Exception(std::string message = "") :
                message(message) {
        }

        std::string what() {
            return message;
        }
    private:
        std::string message;
    };

    struct EmptyChainDataException : public Exception {
        EmptyChainDataException(std::string message = "") :
                Exception(message) {
        }
    };

    struct NoProblemProvidedException : public Exception {
        NoProblemProvidedException(std::string message = "") :
                Exception(message) {
        }
    };

    struct MissingStartingPointsException : public Exception {
        MissingStartingPointsException(std::string message = "") :
                Exception(message) {
        }
    };

    struct UninitializedDataFieldException : public Exception {
        UninitializedDataFieldException(std::string message = "") :
                Exception(message) {
        }
    };
}

#endif // HOPS_EXCEPTIONS_HPP
