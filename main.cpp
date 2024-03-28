#include <iostream>
#include <armadillo>
#include <glog/logging.h>

int main() {
    arma::fmat a;
    int b = 0;
    CHECK(b == 0);
    std::cout << "Hello, World!" << a << std::endl;
    return 0;
}
