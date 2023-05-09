/*
 * 二维矩阵，每一列进行求和，得到一个新矩阵
 */
#include <iostream>

#include "cuda_runtime.h"

const int row = 4;
const int col = 128;


#define GPU_CHECK(ans)                    \
  {                                       \
    GPUAssert((ans), __FILE__, __LINE__); \
  }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
};

__global__ void VecAdd(float *dev_A , float* dev_C){
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float data[col];
    // 拷贝到显存， 各个block读取到各自的block的shared_memory中
    data[tid] = dev_A[bid*col + tid];
    // printf(" data[%.3f]: %.3f", tid, data[tid]);
    __syncthreads();

    // 归约求和
    for(int i = 64; i>0; i /=2){
        if(tid < i){
            data[tid] = data[tid]+ data[tid+i];
        }
        __syncthreads();
    }

    // 取每一个block中的第一个数
    if(tid == 0){
        dev_C[bid] = data[0];
        // printf("dev_C[bid]: %.3f", dev_C[bid]);
    }

}


void CPUVecAdd(float A[row][col] , float B[row]){
    for(int i = 0; i<row; ++i){
        for(int j = 0; j<col; ++j){
            B[i] += A[i][j];
        }
    }
}



int main(){
    float A[row][col], B[row], AA[row*col];
    for(int i = 0; i< row; ++i){
        for(int j = 0; j<col; ++j){
            A[i][j] = (float)(rand()) / RAND_MAX; // 随机小数
            AA[i*col + j] = A[i][j];
        }
    }
    
    // CPU
    CPUVecAdd(A, B);
    std::cout << " --- CPU 结果 ----- "<< std::endl;
    for(int i = 0; i< row ;++i){
        std::cout << B[i] << " , " ;
    }
    std::cout << " 。" << std::endl;
    
    // GPU
    float *dev_A, *dev_C;
    float C[row];
    GPU_CHECK(cudaMalloc(&dev_A, row * col * sizeof(float)));
    GPU_CHECK(cudaMalloc(&dev_C, row * sizeof(float)));
    GPU_CHECK(cudaMemcpy(dev_A, AA, row * col * sizeof(float), cudaMemcpyHostToDevice));
    VecAdd<<<row, col>>>(dev_A, dev_C); // 注意参数必须都是device上的指针
    GPU_CHECK(cudaMemcpy(C, dev_C, row * sizeof(float), cudaMemcpyDeviceToHost));
    GPU_CHECK(cudaDeviceSynchronize());

    std::cout << " --- GPU 结果 ----- "<< std::endl;
    for(int i = 0; i< row ;++i){
        std::cout << C[i] << " , " ;
    }
    std::cout << " 。" << std::endl;

    return 0;
}
