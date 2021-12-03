#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include <complex>
#include <eigen3/Eigen/Dense>

#include "common.h"
#include <iostream>
using namespace std;

class FFT_cpu
{
private:
    int _nx, _ny, _nz;
    ElemType _scale;
    ComplexElemType *_aux4plan;
    #ifdef SINGLE
    fftwf_plan _forward_plan;
    fftwf_plan _backward_plan;
    #else
    fftw_plan _forward_plan;
    fftw_plan _backward_plan;
    #endif

public:

    FFT_cpu(int nx, int ny, int nz)
    {
        _nx = nx;
        _ny = ny;
        _nz = nz;
        _scale = 1.0 / nx / ny / nz;
        int N = nx * ny * nz;
        _aux4plan = new ComplexElemType[N];

        #ifdef SINGLE
        _forward_plan = fftwf_plan_dft_3d(_nx, _ny, _nz, (fftwf_complex *)_aux4plan, (fftwf_complex *)_aux4plan, FFTW_FORWARD, FFTW_ESTIMATE);
        _backward_plan = fftwf_plan_dft_3d(_nx, _ny, _nz, (fftwf_complex *)_aux4plan, (fftwf_complex *)_aux4plan, FFTW_BACKWARD, FFTW_ESTIMATE);
        #else
        _forward_plan = fftw_plan_dft_3d(_nx, _ny, _nz, (fftw_complex *)_aux4plan, (fftw_complex *)_aux4plan, FFTW_FORWARD, FFTW_ESTIMATE);
        _backward_plan = fftw_plan_dft_3d(_nx, _ny, _nz, (fftw_complex *)_aux4plan, (fftw_complex *)_aux4plan, FFTW_BACKWARD, FFTW_ESTIMATE);
        #endif
    }

    ~FFT_cpu()
    {
        delete[] _aux4plan;
        #ifdef SINGLE
        fftwf_destroy_plan(_forward_plan);
        fftwf_destroy_plan(_backward_plan);
        #else
        fftw_destroy_plan(_forward_plan);
        fftw_destroy_plan(_backward_plan);
        #endif
    }

    void fft3d(ComplexVectorType &data, int sign)
    {
        if(sign == FFTW_FORWARD)
        {
            #ifdef SINGLE
            fftwf_execute_dft(_forward_plan, (fftwf_complex *)data.data(), (fftwf_complex *)data.data());
            #else
            fftw_execute_dft(_forward_plan, (fftw_complex *)data.data(), (fftw_complex *)data.data());
            #endif
            data *= _scale;
        }
        else if(sign == FFTW_BACKWARD)
        {
            #ifdef SINGLE
            fftwf_execute_dft(_backward_plan, (fftwf_complex *)data.data(), (fftwf_complex *)data.data());
            #else
            fftw_execute_dft(_backward_plan, (fftw_complex *)data.data(), (fftw_complex *)data.data());
            #endif
        }
    }

    void fft3d_all(ComplexVectorType &data, const ComplexVectorType &vr_eff)
    {
        #ifdef SINGLE
        fftwf_execute_dft(_backward_plan, (fftwf_complex *)data.data(), (fftwf_complex *)data.data());
        data.array() *= vr_eff.array();
        fftwf_execute_dft(_forward_plan, (fftwf_complex *)data.data(), (fftwf_complex *)data.data());
        #else
        fftw_execute_dft(_backward_plan, (fftw_complex *)data.data(), (fftw_complex *)data.data());
        data.array() *= vr_eff.array();
        fftw_execute_dft(_forward_plan, (fftw_complex *)data.data(), (fftw_complex *)data.data());
        #endif
        data *= _scale;
    }
};

#endif
