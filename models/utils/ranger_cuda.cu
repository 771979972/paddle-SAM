#include "paddle/extension.h"


__global__ void RangerKernelREG(float beta1,
                              float beta2,
                              float epsilon,
                              float beta1_pow_,
                              float beta2_pow_,
                              const float* moment1,
                              float* moment1_out,
                              const float* moment2,
                              float* moment2_out,
                              const float* lr_,
                              //const float* step_size_,
                              //const bool* sma_flag_,
                              float step_size,
                              bool sma_flag,
                              float weight_decay,
                              const float* grad,
                              const float* param,
                              float* param_out,
                              int ndim) {
  float lr_orig =  *lr_;
  float beta1_pow = beta1_pow_;
  float beta2_pow = beta2_pow_;
  //float step_size = *step_size_;
  //bool sma_flag = *sma_flag_;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    float p = static_cast<float>(param[id]);
    float g = static_cast<float>(grad[id]);
    float mom1 = moment1[id];
    float mom2 = moment2[id];
    mom1 = beta1 * mom1 + (static_cast<float>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<float>(1.0) - beta2) * g * g;
    
    p -= lr_orig * weight_decay * p;
    if (sma_flag){
      float denom = sqrt(mom2) + epsilon;
      p -= step_size*lr_orig*mom1/denom;
    }else{
      p -= step_size*lr_orig*mom1;
    }

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<float>(p);
  }
}

__global__ void RangerKernelMEM(float beta1,
                              float beta2,
                              float epsilon,
                              const float* beta1_pow_,
                              const float* beta2_pow_,
                              const float* moment1,
                              float* moment1_out,
                              const float* moment2,
                              float* moment2_out,
                              const float* lr_,
                              //const float* step_size_,
                              //const bool* sma_flag_,
                              float step_size,
                              bool sma_flag,
                              float weight_decay,
                              const float* grad,
                              const float* param,
                              float* param_out,
                              int ndim) {
  float lr = *lr_;
  float lr_orig = lr;
  float beta1_pow = *beta1_pow_;
  float beta2_pow = *beta2_pow_;
  //float step_size = *step_size_;
  //bool sma_flag = *sma_flag_;

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    float p = static_cast<float>(param[id]);
    float g = static_cast<float>(grad[id]);
    float mom1 = static_cast<float>(moment1[id]);
    float mom2 = static_cast<float>(moment2[id]);
    mom1 = beta1 * mom1 + (static_cast<float>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<float>(1.0) - beta2) * g * g;

    p -= lr_orig * weight_decay * p;
    if (sma_flag){
      float denom = sqrt(mom2) + epsilon;
      p -= step_size*lr_orig*mom1/denom;
    }else{
      p -= step_size*lr_orig*mom1;
    }

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<float>(p);
  }
}

template <typename T>
__global__ void UpdateBetaPow(T beta1,
                              T beta2,
                              const T* beta1_pow_,
                              const T* beta2_pow_,
                              T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}


std::vector<paddle::Tensor> ranger_cuda_forward(
    // Tensor inputs
    const paddle::Tensor& Param,
    const paddle::Tensor& Grad,
    const paddle::Tensor& LearningRate,
    const paddle::Tensor& Moment1,
    const paddle::Tensor& Moment2,
    const paddle::Tensor& Beta1Pow,
    const paddle::Tensor& Beta2Pow,
    //const paddle::Tensor& StepSize,
    //const paddle::Tensor& SmaFlag,

    // Attrs inputs
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float step_size,
    bool sma_flag
    ) {
  // 输出
  auto ParamOut = paddle::Tensor(paddle::PlaceType::kGPU);
  auto Moment1Out = paddle::Tensor(paddle::PlaceType::kGPU);
  auto Moment2Out = paddle::Tensor(paddle::PlaceType::kGPU);
  auto Beta1PowOut = paddle::Tensor(Beta1Pow.place());
  auto Beta2PowOut = paddle::Tensor(Beta2Pow.place());

  ParamOut.reshape(Param.shape());
  Moment1Out.reshape(Moment1.shape());
  Moment2Out.reshape(Moment2.shape());
  Beta1PowOut.reshape(Beta1Pow.shape());
  Beta2PowOut.reshape(Beta2Pow.shape());

  PD_CHECK(Beta1PowOut.size() == 1,
           "beta1 pow output size should be 1, but received "
           "value is:",
           Beta1PowOut.size());
  PD_CHECK(Beta2PowOut.size() == 1,
           "beta2 pow output size should be 1, but received "
           "value is:",
           Beta2PowOut.size());

  PD_CHECK(Param.type() == paddle::DataType::FLOAT32,
           "Custom ranger support fp32 for now.");

  using T = float;
  auto place = Param.place();

  //不知道是什么，应该 是多线程的部分
  int threads = 512;
  int blocks = (Param.size() + threads - 1) / threads;

  auto Moment1Out_data = Moment1Out.mutable_data<T>(place);
  auto Moment2Out_data = Moment2Out.mutable_data<T>(place);
  auto ParamOut_data = ParamOut.mutable_data<T>(place);

  if (Beta1Pow.place() == paddle::PlaceType::kCPU &&
      Beta2Pow.place() == paddle::PlaceType::kCPU) {
    // CPU:Compute with betapow in REG
    RangerKernelREG<<<blocks, threads, 0, Param.stream()>>>(
        // attr
        beta1,
        beta2,
        epsilon,
        // input
        *Beta1Pow.data<T>(),
        *Beta2Pow.data<T>(),
        Moment1.data<T>(),
        Moment1Out_data,
        Moment2.data<T>(),
        Moment2Out_data,
        LearningRate.data<T>(),
        //StepSize.data<float>(),
        //SmaFlag.data<bool>(),
        step_size,
        sma_flag,
        // attr
        weight_decay,
        // grad and param
        Grad.data<T>(),
        Param.data<T>(),
        ParamOut_data,
        Param.size());
    // Cpu update
    Beta1PowOut.mutable_data<T>(Beta1Pow.place())[0] =
        beta1 * Beta1Pow.data<T>()[0];
    Beta2PowOut.mutable_data<T>(Beta2Pow.place())[0] =
        beta2 * Beta2Pow.data<T>()[0];
  } else {
    // GPU:Compute with betapow in MEM
    RangerKernelMEM<<<blocks, threads, 0, Param.stream()>>>(
        beta1,
        beta2,
        epsilon,
        Beta1Pow.data<T>(),
        Beta2Pow.data<T>(),
        Moment1.data<T>(),
        Moment1Out_data,
        Moment2.data<T>(),
        Moment2Out_data,
        LearningRate.data<T>(),
        //StepSize.data<float>(),
        //SmaFlag.data<bool>(),
        step_size,
        sma_flag,
        weight_decay,
        Grad.data<T>(),
        Param.data<T>(),
        ParamOut_data,
        int(Param.size()));
    // Update with gpu
    UpdateBetaPow<T><<<1, 32, 0, Param.stream()>>>(
        beta1,
        beta2,
        Beta1Pow.data<T>(),
        Beta2Pow.data<T>(),
        Beta1PowOut.mutable_data<T>(place),
        Beta2PowOut.mutable_data<T>(place));
  }

  return {ParamOut, Moment1Out, Moment2Out, Beta1PowOut, Beta2PowOut};
}