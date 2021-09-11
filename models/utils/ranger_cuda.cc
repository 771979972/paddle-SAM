#include "paddle/extension.h"
#include <vector>
#include <iostream>

std::vector<paddle::Tensor> ranger_cuda_forward(
    // tensor inputs
    const paddle::Tensor& Param,
    const paddle::Tensor& Grad,
    const paddle::Tensor& LearningRate,
    const paddle::Tensor& Moment1,
    const paddle::Tensor& Moment2,
    const paddle::Tensor& Beta1Pow,
    const paddle::Tensor& Beta2Pow,
    //const paddle::Tensor& StepSize,
    //const paddle::Tensor& SmaFlag,

    // attrs inputs
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float step_size,
    bool sma_flag
    );

std::vector<paddle::Tensor> RangerForward(
    // tensor inputs
    const paddle::Tensor& Param,
    const paddle::Tensor& Grad,
    const paddle::Tensor& LearningRate,
    const paddle::Tensor& Moment1,
    const paddle::Tensor& Moment2,
    const paddle::Tensor& Beta1Pow,
    const paddle::Tensor& Beta2Pow,
    //const paddle::Tensor& StepSize,
    //const paddle::Tensor& SmaFlag,

    // attrs inputs
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float step_size,
    bool sma_flag
    ){
        if (Param.place()==paddle::PlaceType::kCPU){
            PD_THROW("Not implemented.");
        } else if (Param.place() == paddle::PlaceType::kGPU){
            return ranger_cuda_forward(
                Param, Grad, LearningRate, Moment1, Moment2, Beta1Pow, Beta2Pow, //StepSize, SmaFlag,
                beta1, beta2, epsilon, weight_decay, step_size, sma_flag
            );
        } else {
            PD_THROW("Not implemented.");
        }
    }

// 输入输出形状
std::vector<std::vector<int64_t>> RangerInferShape(
    std::vector<int64_t> param_shape,
    std::vector<int64_t> grad_shape,
    std::vector<int64_t> lr_shape,
    std::vector<int64_t> m1_shape,
    std::vector<int64_t> m2_shape,
    std::vector<int64_t> b1_shape,
    std::vector<int64_t> b2_shape
    //std::vector<int64_t> step_shape,
    //std::vector<int64_t> flag_shape
    ) {
        return {param_shape, m1_shape, m2_shape, b1_shape, b2_shape};
    }

// 输入输出类型
std::vector<paddle::DataType> RangerInferDtype(
    paddle::DataType param_dtype,
    paddle::DataType grad_dtype,
    paddle::DataType lr_dtype,
    paddle::DataType m1_dtype,
    paddle::DataType m2_dtype,
    paddle::DataType b1_dtype,
    paddle::DataType b2_dtype
    //paddle::DataType step_dtype,
    //paddle::DataType flag_dtype
    ) {
  return {param_dtype, m1_dtype, m2_dtype, b1_dtype, b2_dtype};
}

// input, output, attr
PD_BUILD_OP(ranger)
    .Inputs({
        "Param",         // "(Tensor) Input parameter"
        "Grad",          // "(Tensor) Input gradient"
        "LearningRate",  // "(Tensor) Learning rate"
        "Moment1",       // "(Tensor) Input first moment"
        "Moment2",       // "(Tensor) Input second moment"
        "Beta1Pow",      // "(Tensor) Input beta1 power accumulator"
        "Beta2Pow"      // "(Tensor) Input beta2 power accumulator"
        //"StepSize",
        //"SmaFlag"
    })
    .Outputs({
        "ParamOut",     //  "(Tensor) Output parameter");
        "Moment1Out",   //  "(Tensor) Output first moment");
        "Moment2Out",   //  "(Tensor) Output second moment");
        "Beta1PowOut",  //  "(Tensor) Output beta1 power accumulator");
        "Beta2PowOut"  //  "(Tensor) Output beta2 power accumulator");
    })
    .Attrs({
        "beta1: float",  
        "beta2: float", 
        "epsilon: float",
        "weight_decay: float",
        "step_size: float",
        "sma_flag: bool"
    })
    .SetKernelFn(PD_KERNEL(RangerForward))
    .SetInferShapeFn(PD_INFER_SHAPE(RangerInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RangerInferDtype));