#pragma once
#include "complex_utils.h"
#include "common.h"
#include <cufft.h>
namespace faster_dft
{

template<OperationType OpType>
class FFT
{
private:
    typedef Traits<OpType> Traits_;
    typedef typename Traits_::ComplexType ComplexType;
    typedef typename Traits_::ElemType ElemType;
    cufftHandle plan_;
    const int nx_, ny_, nz_, N_;
	const int batch_;
public:
    FFT(const int nx, const int ny, const int nz, const int batch):
        nx_(nx), ny_(ny), nz_(nz), N_(nx * ny * nz), batch_(batch)
    {
		cufftType type = OpType == OperationType::FP64 ? CUFFT_Z2Z : CUFFT_C2C;
		if(batch_ == 1)
		{
			cufftPlan3d(&plan_, nx, ny, nz, type);
		}
		else
		{
			int n[3] = {nx_, ny_, nz_};
			cufftPlanMany(&plan_, 3, n, 
					NULL, 1, N_, // *inembed, istride, idist 
					NULL, 1, N_, // *onembed, ostride, odist
					type, batch_);
		}
    }

    void fft3d_all(const ElemType *vr_eff, ComplexType *data, cudaStream_t stream);
    ~FFT()
    {
        //destroy handle
        cufftDestroy(plan_);
    }
};

}
