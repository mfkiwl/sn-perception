#include <iostream>
#include <Eigen/Dense>


// slides: https://epcced.github.io/2019-04-16-ModernCpp/lectures/eigen/using-eigen.pdf


int main() {
    Eigen::Matrix<double, 10, 10> A;
    A.setZero();
    A(9, 0) = 1.234;

    std::cout << A << std::endl;
    
    return 0;
}