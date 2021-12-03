/*
* Author: Xiaoying Jia
* Project: FasterDFT
* Department: ByteDance Data-AML
* Email: {jiaxiaoying}@bytedance.com
*/
#include "../includes/eigen_solver.h"
#include "../includes/orth.h"
#include "../includes/common.h"
#include "../includes/reduce.h"
#include "../unit_test/helper.h"
#include <cmath>
using namespace std;

#define WARP_SIZE  32
#define PI 3.141592653589793

namespace faster_dft
{

template <typename T>
void print_double(const T *data, const int size, const char *ss)
{
    T *h_data = (T *)malloc(sizeof(T) * size);
    cudaMemcpy(h_data, data, sizeof(T) * size, cudaMemcpyDeviceToHost);
    printf("%s values\n", ss);
    for(int i = 0; i < size; ++i)
        printf("%d %lf\n", i, h_data[i]);
    printf("\n");
    free(h_data);
}

template <typename T>
__global__
void dot_kernel(const T *X, const T *Y, T *dot_val, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    T val;
    val.x = 0.0;
    val.y = 0.0;
    if(tid < N)
    {
        T a = __ldg(&X[tid]);
        T b = __ldg(&Y[tid]);
        //a dot conjugate of b
        val.x = a.x * b.x + a.y * b.y;
        val.y = a.x * b.y - a.y * b.x;
    }

    val.x = blockReduceSum(val.x);
    val.y = blockReduceSum(val.y);

    __syncthreads();
    if(threadIdx.x == 0)
    {
        atomicAdd(&dot_val[0].x, val.x);
        atomicAdd(&dot_val[0].y, val.y);
    }
}

template <typename ComplexType, typename ElemType>
__global__
void dot_kernel(const ComplexType *X, const ComplexType *Y, ElemType *dot_val, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    ComplexType val;
    val.x = 0.0;
    if(tid < N)
    {
        ComplexType a = __ldg(&X[tid]);
        ComplexType b = __ldg(&Y[tid]);

        //a dot conjugate of b
        val.x = a.x * b.x + a.y * b.y;
    }
    val.x = blockReduceSum(val.x);
    if(threadIdx.x == 0)
    {
        atomicAdd(&dot_val[0], val.x);
    }
}

//gg_buf [inter, now, last]
/*
   ElemType gg_inter = dot_real(g, Pg);
   Pg = P.apply(g);
   ElemType gg_now = dot_real(g, Pg);
   if(iter == 0)
   {
    gg_last = gg_now;
    cg = g;
   }
 */

template <typename ComplexType, typename ElemType>
__global__
void cal_gg_inter_now_kernel(const ComplexType *g, const ElemType *pre_conditioner, ElemType *gg_buf, ElemType *cg_norm,
                             ComplexType *Pg, ComplexType *cg, const int iter, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    ComplexType old_Pg = __ldg(&Pg[tid]);
    ComplexType g_val;
    ComplexType Pg_val;

    ElemType gg_now = 0.0;
    ElemType gg_inter = 0.0;

    if(tid < N)
    {
        g_val = __ldg(&g[tid]);
        Pg_val.x = g_val.x * pre_conditioner[tid];
        Pg_val.y = g_val.y * pre_conditioner[tid];
        gg_now = g_val.x * Pg_val.x + g_val.y * Pg_val.y;
        Pg[tid] = Pg_val;
    }
    gg_now = blockReduceSum(gg_now);
    if(iter == 0)
    {
        ElemType cg_norm_val = 0.0;
        if(tid < N)
            cg_norm_val = g_val.x * g_val.x + g_val.y * g_val.y;

        cg_norm_val = blockReduceSum(cg_norm_val);

        if(tid < N)
            cg[tid] = g_val;

        //gg_last = gg_now
        //update cg_norm
        if(threadIdx.x == 0)
        {
            atomicAdd(&gg_buf[2], gg_now);
            atomicAdd(&cg_norm[0], cg_norm_val);
        }
    }
    else
    {
        //update gg_inter
        gg_inter = g_val.x * old_Pg.x + g_val.y * old_Pg.y;
        gg_inter = blockReduceSum(gg_inter);
        if(threadIdx.x == 0)
        {
            atomicAdd(&gg_buf[1], gg_now);
            atomicAdd(&gg_buf[0], gg_inter);
        }
    }
}


template <typename ComplexType, typename ElemType>
ElemType dot_real_kernel_launcher(const ComplexType *X, const ComplexType *Y, ElemType *dot_val, const int N, cudaStream_t stream)
{
    cudaMemsetAsync(dot_val, 0, sizeof(ElemType), stream);

    dim3 block(128);
    dim3 grid((N - 1) / block.x + 1);

    dot_kernel<<<grid, block, 0, stream>>>(X, Y, dot_val, N);

    ElemType cpu_dot_val;
    cudaMemcpyAsync(&cpu_dot_val, dot_val, sizeof(ElemType), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return cpu_dot_val;
}

template <typename ComplexType, typename ElemType>
__global__
void update_cg_kernel(const ComplexType *x, const ComplexType *g, ComplexType *cg, ElemType *gg_buf, ElemType *cg_norm,
                      const ElemType old_cg_norm, const ElemType theta, const int N)
{

    /*
       const ElemType gamma = (gg_now - gg_inter) / gg_last;
       gg_last = gg_now;
       cg = cg * gamma + g;

       const ElemType norma = gamma * cg_norm * sin(theta);
       cg -= norma * x;

    */
    /*
        [inter, now, last]
    */
    ElemType gg_now = gg_buf[1];
    ElemType gg_inter = gg_buf[0];
    ElemType gg_last = gg_buf[2];
    ElemType gamma = (gg_now - gg_inter) / gg_last;
    ElemType norma = gamma * old_cg_norm * sin(theta);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    ElemType cg_norm_val = 0.0;
    if(tid < N)
    {
        ComplexType cg_val = cg[tid];
        ComplexType g_val = __ldg(&g[tid]);
        ComplexType x_val = __ldg(&x[tid]);
        cg_val.x = cg_val.x * gamma + g_val.x - norma * x_val.x;
        cg_val.y = cg_val.y * gamma + g_val.y - norma * x_val.y;
        cg[tid] = cg_val;
        cg_norm_val = cg_val.x * cg_val.x + cg_val.y * cg_val.y;
    }
    cg_norm_val = blockReduceSum(cg_norm_val);
    if(threadIdx.x == 0)
        atomicAdd(&cg_norm[0], cg_norm_val);
    if(tid == 0)
    {
        //update gg_last = gg_now
        gg_buf[2] = gg_now;
    }
}

template <typename ComplexType, typename ElemType>
__global__
void dot_x_PHx_Px_kernel(const ComplexType *X_ptr, const ComplexType *Hx_ptr, const ElemType *P_ptr, ElemType *gpu_tmp, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    ComplexType zero_complex;
    zero_complex.x = (ElemType)0.0;
    zero_complex.y = (ElemType)0.0;

    ComplexType x = tid < N ? __ldg(&X_ptr[tid]) : zero_complex;
    ComplexType Hx = tid < N ? __ldg(&Hx_ptr[tid]) : zero_complex;
    ElemType p_inv = tid < N ? (ElemType)(1.0) / __ldg(&P_ptr[tid]) : 0.0;

    ComplexType px = zero_complex;
    ComplexType pHx = zero_complex;

    //Px = x.cwiseProduct(P.cwiseInverse());
    //PHx = Hx.cwiseProduct(P.cwiseInverse());
    //g = PHx - (dot_real(x, PHx) / dot_real(x, Px)) * Px;
    ElemType x_phx = 0.0;
    ElemType x_px = 0.0;
    if(tid < N)
    {
        px.x = x.x * p_inv;
        px.y = x.y * p_inv;
        pHx.x = Hx.x * p_inv;
        pHx.y = Hx.y * p_inv;

        x_phx = x.x * pHx.x + x.y * pHx.y;
        x_px = x.x * px.x + x.y * px.y;
    }
    x_phx = blockReduceSum(x_phx);
    x_px = blockReduceSum(x_px);

    if(threadIdx.x == 0)
    {
        atomicAdd(&gpu_tmp[0], x_phx);
        atomicAdd(&gpu_tmp[1], x_px);
    }
}
template <typename ComplexType, typename ElemType>
__global__
void cal_g_kernel(const ComplexType *X_ptr, const ComplexType *Hx_ptr, const ElemType *P_ptr, const ElemType *gpu_tmp, ComplexType *g_ptr, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    ComplexType zero_complex;
    zero_complex.x = (ElemType)0.0;
    zero_complex.y = (ElemType)0.0;

    ComplexType x = tid < N ? __ldg(&X_ptr[tid]) : zero_complex;
    ComplexType Hx = tid < N ? __ldg(&Hx_ptr[tid]) : zero_complex;
    ElemType p_inv = tid < N ? (ElemType)(1.0) / __ldg(&P_ptr[tid]) : 0.0;

    ElemType scaler = gpu_tmp[0] / gpu_tmp[1];

    ComplexType px = zero_complex;
    ComplexType pHx = zero_complex;
    ComplexType g = zero_complex;

    //Px = x.cwiseProduct(P.cwiseInverse());
    //PHx = Hx.cwiseProduct(P.cwiseInverse());
    //g = PHx - (dot_real(x, PHx) / dot_real(x, Px)) * Px;
    if(tid < N)
    {
        px.x = x.x * p_inv;
        px.y = x.y * p_inv;
        pHx.x = Hx.x * p_inv;
        pHx.y = Hx.y * p_inv;

        g.x = pHx.x - scaler * px.x;
        g.y = pHx.y - scaler * px.y;
        g_ptr[tid] = g;
    }
}



template <typename ComplexType, typename ElemType>
void cal_g_kernel_launcher(const ComplexType *X, const ComplexType *Hx, const ElemType *P, ElemType *gpu_tmp, ComplexType *g,
                           const int N, cudaStream_t stream)
{

    //Px = x.cwiseProduct(P.cwiseInverse());
    //PHx = Hx.cwiseProduct(P.cwiseInverse());
    //g = PHx - (dot_real(x, PHx) / dot_real(x, Px)) * Px;
    cudaMemsetAsync(gpu_tmp, 0, sizeof(ElemType) * 2, stream);

    dim3 block(128);
    dim3 grid((N - 1) / block.x + 1);

    //Px = x / P
    //PHx = Hx / P
    //gpu_tmp[0] = x * PHx, gpu_tmp[1] = x * Px

    dot_x_PHx_Px_kernel<ComplexType, ElemType><<<grid, block, 0, stream>>>(X, Hx, P, gpu_tmp, N);
    //g = PHx - (dot_real(x, PHx) / dot_real(x, Px)) * Px;
    cal_g_kernel<<<grid, block, 0, stream>>>(X, Hx, P, gpu_tmp, g, N);
}

//gg_buf [inter, now, last]
template <typename ComplexType, typename ElemType>
ElemType update_cg_kernel_launcher(const ComplexType *x, const ComplexType *g,
                                   const ElemType *pre_conditioner, ElemType *gg_buf, ElemType *cg_norm,
                                   ComplexType *Pg, ComplexType *cg, const ElemType old_cg_norm, const ElemType theta,
                                   const int iter, const int N, cudaStream_t stream)
{
    /*
    ElemType gg_inter = dot_real(g, Pg);
    Pg = P.apply(g);
    ElemType gg_now = dot_real(g, Pg);
    if(iter == 0)
    {
        gg_last = gg_now;
        cg = g;
    }
    else
    {
        const ElemType gamma = (gg_now - gg_inter) / gg_last;
        gg_last = gg_now;
        cg = cg * gamma + g;
        const ElemType norma = gamma * cg_norm * sin(theta);
        cg -= norma * x;
    }
    */
    ElemType norm_val;
    dim3 block(128);
    dim3 grid((N - 1) / block.x + 1);

    cudaMemsetAsync(gg_buf, 0, sizeof(ElemType) * 2, stream);
    cudaMemsetAsync(cg_norm, 0, sizeof(ElemType), stream);

    cal_gg_inter_now_kernel<<<grid, block, 0, stream>>>(g, pre_conditioner, gg_buf, cg_norm, Pg, cg, iter, N);

    //print_complex(Pg, 10, "Pg");
    //print_double(gg_buf, 3, "gg inter/now/last");

    if(iter > 0)
        update_cg_kernel<<<grid, block, 0, stream>>>(x, g, cg, gg_buf, cg_norm, old_cg_norm, theta, N);
    cudaMemcpyAsync(&norm_val, cg_norm, sizeof(ElemType), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    norm_val = sqrt(norm_val);
    return norm_val;
}

template <typename ComplexType, typename ElemType>
__global__
void update_x_kernel(const ComplexType *cg, ComplexType *x, ElemType *norm_val,
                     const ElemType cost, const ElemType sint_norm, const bool norm_x, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    ElemType x_norm_val = 0.0;

    if(tid < N)
    {
        ComplexType x_val = x[tid];
        ComplexType cg_val = __ldg(&cg[tid]);

        x_val.x = x_val.x * cost + cg_val.x * sint_norm;
        x_val.y = x_val.y * cost + cg_val.y * sint_norm;

        x[tid] = x_val;

        if(norm_x)
            x_norm_val = x_val.x * x_val.x + x_val.y * x_val.y;
    }

    if(!norm_x)
        return;

    x_norm_val = blockReduceSum(x_norm_val);
    if(threadIdx.x == 0)
        atomicAdd(&norm_val[0], x_norm_val);
}

template <typename ComplexType, typename ElemType>
__global__
void update_Hx_x_kernel(const ComplexType *Hcg, ComplexType *Hx, ComplexType *x,
                        const ElemType *x_norm_val, const ElemType cost, const ElemType sint_norm,
                        const bool norm_x, const bool update_Hx, const int N)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid >= N)
        return;

    if(norm_x)
    {
        ElemType norm_val = 1.0 / __ldg(&x_norm_val[0]);
        x[tid].x *= norm_val;
        x[tid].y *= norm_val;
    }
    if(update_Hx)
    {
        ComplexType Hx_val = Hx[tid];
        ComplexType Hcg_val = __ldg(&Hcg[tid]);
        Hx_val.x = Hx_val.x * cost + Hcg_val.x * sint_norm;
        Hx_val.y = Hx_val.y * cost + Hcg_val.y * sint_norm;
        Hx[tid] = Hx_val;
    }
}

template <typename ComplexType, typename ElemType>
void update_x_Hx(const ComplexType *Hcg, const ComplexType *cg, ComplexType *x, ComplexType *Hx,
                 ElemType *ElemType_gpu_tmp, const ElemType cost,
                 const ElemType sint_norm, const bool norm_x, const bool update_Hx, const int N, cudaStream_t stream)
{

    dim3 block(128);
    dim3 grid((N - 1) / block.x + 1);

    cudaMemsetAsync(ElemType_gpu_tmp, 0, sizeof(ElemType), stream);

    update_x_kernel<ComplexType, ElemType><<<grid, block, 0, stream>>>(cg, x, ElemType_gpu_tmp, cost, sint_norm, norm_x, N);
    update_Hx_x_kernel<ComplexType, ElemType><<<grid, block, 0, stream>>>(Hcg, Hx, x,
            ElemType_gpu_tmp, cost, sint_norm, norm_x, update_Hx, N);
}

template <typename ComplexType, typename ElemType>
__global__
void error_kernel(const ComplexType *Hx, const ComplexType *x, ElemType *gpu_buf, const ElemType lambda, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    ElemType tmp = 0.0;
    if(tid < N)
    {
        ElemType a = Hx[tid].x - lambda * x[tid].x;
        ElemType b = Hx[tid].y - lambda * x[tid].y;
        tmp += sqrt(a * a + b * b);
    }
    tmp = blockReduceSum(tmp);
    if(threadIdx.x == 0)
    {
        atomicAdd(&gpu_buf[0], tmp);
    }
}
template <typename ComplexType, typename ElemType>
ElemType error_kernel_launcher(const ComplexType *Hx, const ComplexType *x, ElemType *gpu_buf, const ElemType lambda, const int N,
                               cudaStream_t stream)
{
    /*
    ElemType error(const ComplexVectorType & Hx, const ComplexVectorType & x, const ElemType lambda) {
        return (Hx - lambda * x).cwiseAbs().mean();
    }

    */
    ElemType cpu_val;
    cudaMemsetAsync(gpu_buf, 0, sizeof(ElemType), stream);

    dim3 block(128);
    dim3 grid((N -1)/ block.x + 1);
    error_kernel<<<grid, block, 0, stream>>>(Hx, x, gpu_buf, lambda, N);
    cudaMemcpyAsync(&cpu_val, gpu_buf, sizeof(ElemType), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return cpu_val / N;
}
//X K * N
template <OperationType OpType>
void EigenSolver<OpType>::solve(ComplexType *X, ElemType *he, void *buf,
                                const int K, cublasHandle_t cublas_handle, cudaStream_t stream)
{
    OrthParam<OpType> orth_param{X, x_col, orth_buf, K, N_, false, cublas_handle, stream};

    int cnt = 0;
    memset(he, 0, sizeof(ElemType) * K);
    //reset_buf(stream);

    for(int m = 0; m < K; ++m)
    {
        //printf("m %d\n", m);
        check_cuda_error(cudaMemcpyAsync(x_col, X + m * N_, sizeof(ComplexType) * N_, cudaMemcpyDeviceToDevice, stream));
        //print_complex(x_col, 10, "x_col");
        orth_param.V = x_col;
        orth_param.K = m;
        orth_param.norm = true;
        //orth(X, m, x);
        orth_kernel_launcher(orth_param);
        //print_complex(x_col, 10, "after orth x");

        //Hx = H.apply(x)
        hamiltion_layer_->apply(x_col, Hx, (void *)(NULL), cublas_handle, stream);
        //print_complex(Hx, 10, "Hx");

        he[m] = dot_real_kernel_launcher<ComplexType, ElemType>(x_col, Hx, double_gpu_tmp, N_, stream);

        //printf("he %lf\n", he[m]);

        ElemType theta = 0.0;
        ElemType cg_norm = 0.0;

        cudaMemsetAsync(gg_buf, 0, sizeof(ElemType) * 3, stream);

        for(int iter = 0; iter < max_ite_; ++iter)
        {
            if(random_init_)
            {
                ElemType err = error_kernel_launcher(Hx, x_col, double_gpu_tmp, he[m], N_, stream);
                printf("m %d iter %d err %lf\n", m, iter, err);
                if(err < eps_)
                    break;
            }
            cnt++;
            cal_g_kernel_launcher(x_col, Hx, pre_conditioner_, double_gpu_tmp, g, N_, stream);
            orth_param.X = X;
            orth_param.V = g;
            orth_param.K = m;
            orth_param.norm = false;
            orth_kernel_launcher(orth_param);
            cg_norm = update_cg_kernel_launcher(x_col, g, pre_conditioner_, gg_buf, double_gpu_tmp,
                                                Pg, cg, cg_norm, theta,
                                                iter, N_, stream);

            hamiltion_layer_->apply(cg, Hcg, (void *)(NULL), cublas_handle, stream);

            //printf("m %d iter %d cg_norm %lf\n", m, iter, cg_norm);
            if(!random_init_)
            {
                if(cg_norm < eps_)
                    break;
            }

            ElemType a0 = dot_real_kernel_launcher(x_col, Hcg, double_gpu_tmp, N_, stream) * 2.0 / cg_norm;
            ElemType b0 = dot_real_kernel_launcher(cg, Hcg, double_gpu_tmp, N_, stream) / cg_norm / cg_norm;

            ElemType e0 = he[m];
            theta = atan(a0 / (e0 - b0)) * 0.5;

            ElemType new_e = (e0 - b0) * cos(2.0 * theta) + a0 * sin(2.0 * theta);
            ElemType e1 = (e0 + b0 + new_e) * 0.5;
            ElemType e2 = (e0 + b0 - new_e) * 0.5;
            if(e1 > e2)
            {
                theta += 1.5707963;
            }
            he[m] = std::min(e1, e2);
            ElemType cost = cos(theta);
            ElemType sint_norm = sin(theta) / cg_norm;

            bool stop_flag = (fabs(he[m] - e0) < eps_) ? true : false;

            update_x_Hx(Hcg, cg, x_col, Hx, double_gpu_tmp, cost, sint_norm, iter > 0, !stop_flag, N_, stream);

            if(stop_flag)
                break;
        }

        orth_param.V = x_col;
        orth_param.K = m;
        orth_param.norm = true;
        orth_kernel_launcher(orth_param);
        //update X

        //printf("m %d he %lf\n", m, he[m]);
        if(m > 0 && he[m] < he[m-1])
        {
            //printf("change order\n");
            int insertIdx = 0;
            for(int i = 0; i < m; ++i)
            {
                if(he[m] < he[i])
                {
                    insertIdx = i;
                    break;
                }
            }
            //printf("m %d change order insertIdx %d\n", m, insertIdx);
            ElemType em = he[m];
            for(int i = m; i > insertIdx; --i)
                he[i] = he[i-1];
            he[insertIdx] = em;
            //[insertId, m - 1] -> X_buf
            cudaMemcpyAsync(X_buf, X + insertIdx * N_, sizeof(ComplexType) * (m - insertIdx) * N_, cudaMemcpyDeviceToDevice, stream);
            //x_col ->X insertId
            cudaMemcpyAsync(X + insertIdx * N_, x_col, sizeof(ComplexType) * N_, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(X + (insertIdx + 1) * N_, X_buf, sizeof(ComplexType) * N_ * (m - insertIdx), cudaMemcpyDeviceToDevice, stream);
            //insert x col
        }
        else
        {
            cudaMemcpyAsync(X + m * N_, x_col, sizeof(ComplexType) * N_, cudaMemcpyDeviceToDevice, stream);
        }
    }
    printf("cnt %d\n", cnt);
}
template void EigenSolver<OperationType::FP64>::solve(cuDoubleComplex *X, double *eigen_values, void *buf,
        const int K, cublasHandle_t handle, cudaStream_t stream);
template void EigenSolver<OperationType::FP32>::solve(cuComplex *X, float *eigen_values, void *buf,
        const int K, cublasHandle_t handle, cudaStream_t stream);

}

