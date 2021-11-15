#ifndef HOPS_PROBLEM_HPP
#define HOPS_PROBLEM_HPP

#include <Eigen/Core>

#include <cassert>

namespace hops {
    template<typename Model, typename Proposal>
    class RunBase;

    template<typename Model>
    class Problem {
    public:
        using ModelType = Model;

        //Problem() = default;

        Problem(const Model& model) :
                model(model) {
        }

        Problem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) :
                A(A),
                b(b) {
            dimension = A.cols();
        }

        Problem(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Model& model) :
                A(A),
                b(b),
                model(model) {
            dimension = A.cols();
        }

        long getDimension() {
            return dimension;
        }

        void setA(const Eigen::MatrixXd& A) {
            this->A = A;
            this->dimension = A.cols();
        }

        void setB(const Eigen::VectorXd& b) {
            this->b = b;
        }

        const Eigen::MatrixXd& getA() const {
            return this->A;
        }

        const Eigen::VectorXd& getB() const {
            return this->b;
        }

        const Model& getModel() const {
            return this->model;
        }

        void setStartingPoint(const Eigen::VectorXd& startingPoint) {
            if (startingPoint.rows() > 0) {
                this->startingPoint = startingPoint;
                useStartingPoint = true;
            } else {
                this->startingPoint = startingPoint;
                useStartingPoint = false;
            }
        }

        const Eigen::VectorXd& getStartingPoint() {
            return this->startingPoint;
        }

        void setUnroundingTransformation(const Eigen::MatrixXd& unroundingTransformation) {
            if (unroundingTransformation.size() > 0) {
                this->unroundingTransformation = unroundingTransformation;
                unround = true;
            } else {
                this->unroundingTransformation = unroundingTransformation;
                unround = false;
            }
        }

        const Eigen::MatrixXd& getUnroundingTransformation() {
            return this->unroundingTransformation;
        }

        void setUnroundingShift(const Eigen::MatrixXd& unroundingShift) {
            if (unroundingShift.size() > 0) {
                this->unroundingShift = unroundingShift;
                unround = true;
            } else {
                this->unroundingShift = unroundingShift;
                unround = false;
            }
        }

        const Eigen::VectorXd& getUnroundingShift() {
            return this->unroundingShift;
        }

    private:
        Eigen::MatrixXd A;
        Eigen::VectorXd b;

        long dimension;

        Model model;

        bool unround = false;
        Eigen::MatrixXd unroundingTransformation;
        Eigen::VectorXd unroundingShift;

        bool useStartingPoint = false;
        Eigen::VectorXd startingPoint;

        template<typename, typename> friend class RunBase;
    };
}

#endif // HOPS_PROBLEM_HPP
