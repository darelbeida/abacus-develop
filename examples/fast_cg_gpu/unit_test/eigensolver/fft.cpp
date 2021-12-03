#include <iostream>

#include "fft.h"

int main(int argc, char ** argv) {
    int n1 = 32;
    int n2 = 32;
    int n3 = 32;
    FFT fft(n1, n2, n3);
    

    ComplexVectorType input = ComplexVectorType::Random(n1 * n2 * n3);
    ComplexVectorType output = fft.fft3d(input, FFTW_FORWARD);

    std::cout << input << std::endl
        << "--------" << std::endl
        << output << std::endl;
    return 0;
}