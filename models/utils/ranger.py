import os
import paddle
import math
from paddle.optimizer.optimizer import Optimizer
from collections import defaultdict
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.framework import Variable
from paddle.fluid import layers
from paddle.fluid import unique_name
from paddle.fluid.framework import in_dygraph_mode, _dygraph_tracer
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import base as imperative_base
import numpy as np
from paddle.utils.cpp_extension import get_build_directory

# op
build_dir = get_build_directory()
op_lib = os.path.join(build_dir, "ranger_op/ranger_op.so")
if op_lib is not None and os.path.isfile(op_lib):
    # Maybe it has been loadad by `ext_utils.load`
    paddle.utils.cpp_extension.load_op_meta_info_and_register_op(
        op_lib)

class Ranger(Optimizer):
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"
    _slow_str = "slow"
    #_inf_norm_acc_str = "inf_norm"

    def __init__(self,
                 learning_rate=0.001,
                 alpha=0.5, k=6, # Look Ahead
                 beta1=0.95, beta2=0.999, epsilon=1e-5, weight_decay=0.0, N_sma_threshhold=5.,  # RAdam
                 use_gc=True,gc_conv_only=False, # gradient centralization
                 parameters=None,
                 name=None):
        if not isinstance(beta1, Variable):
            if not 0 <= beta1 < 1:
                raise ValueError(
                    "Invaild value of beta1, expect beta1 in [0,1).")
        if not isinstance(beta2, Variable):
            if not 0 <= beta2 < 1:
                raise ValueError(
                    "Invaild value of beta2, expect beta2 in [0,1).")
        if not isinstance(epsilon, Variable):
            if not 0 <= epsilon:
                raise ValueError(
                    "Invaild value of epsilon, expect epsilon >= 0.")
        assert (
            0.0 <= alpha <= 1.0
        ), "alpha should be larger or equal to 0.0, and less or equal than 1.0"
        assert (isinstance(k, int) and k > 0), "k should be a positive integer"

        super(Ranger, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=None,
            name=name)
        self.type = "ranger"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._N_sma_threshhold = N_sma_threshhold
        self._N_sma_max =  2 / (1 - beta2) - 1 # ρ无穷

        self.use_gc = use_gc
        self.gc_gradient_threshold = 3 if gc_conv_only else 1
        self.alpha = alpha
        self.k = k
        self.helper = LayerHelper(self.__class__.__name__)

        #self._k_var = None
        #self._global_step_var = None
        #self._step_size_var = None
        #self._sma_flag_var = None
        self._global_step = 0
        self._step_size = None
        self._sma_flag = None


    def _get_accumulator(self, name, param):
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def _add_moments_pows(self, p):
        self._add_accumulator(self._moment1_acc_str, p)
        self._add_accumulator(self._moment2_acc_str, p)
        #self._add_accumulator(self._inf_norm_acc_str, p)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            fill_value=self._beta1,
            shape=[1])
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            fill_value=self._beta2,
            shape=[1])
    """
    def _increment_global_var(self):
        # 如果不行的话，把加一放到c文件里面
        if self._global_step_var is None:
            self._global_step_var = layers.create_global_var(
                name=unique_name.generate("lookhead_step"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True
            )
        
        self.helper.append_op(
            type='increment',
            inputs={'X': [self._global_step_var]},
            outputs={'Out':[self._global_step_var]},
            attrs={'step': 1.0}
        )
    """

    def _cal_sma(self):
        """      
        beta2_pow = self._beta2**self._global_step_var
        beta1_pow = self._beta1**self._global_step_var
        N_sma = self._N_sma_max - 2. * self._global_step_var * beta2_pow / (1. - beta2_pow)
        
        sma_flag = N_sma > self._N_sma_threshhold

        if sma_flag:
            step_size = paddle.sqrt( (1.-beta2_pow) * (N_sma-4.) / (self._N_sma_max-4.) * (N_sma-2.) / N_sma * self._N_sma_max /(self._N_sma_max-2.)) / (1.-beta1_pow)
        else:
            step_size = 1./(1. - beta1_pow)
        if self._step_size_var is None:
            self._step_size_var = layers.create_global_var(
                name=unique_name.generate("radam_step_size"),
                shape=[1],
                value=step_size,
                dtype='int32',
                persistable=True
            )
        if self._sma_flag_var is None:
            self._sma_flag_var = layers.create_global_var(
                name=unique_name.generate("radam_sma_flag"),
                shape=[1],
                value=sma_flag,
                dtype='bool',
                persistable=True
            )
        
        paddle.assign(step_size, self._step_size_var)
        paddle.assign(sma_flag, self._sma_flag_var)
        """ 
        beta2_pow = self._beta2**self._global_step
        beta1_pow = self._beta1**self._global_step
        N_sma = self._N_sma_max - 2. * self._global_step * beta2_pow / (1. - beta2_pow)
        
        sma_flag = N_sma > self._N_sma_threshhold

        if sma_flag:
            step_size = math.sqrt( (1.-beta2_pow) * (N_sma-4.) / (self._N_sma_max-4.) * (N_sma-2.) / N_sma * self._N_sma_max /(self._N_sma_max-2.)) / (1.-beta1_pow)
        else:
            step_size = 1./(1. - beta1_pow)
        
        self._step_size = step_size
        self._sma_flag = sma_flag

    def _append_optimize_op(self, block, param_and_grad):
        # gradient centralization对于grad，不是param
        # GC operation for Conv layers and FC layers
        # GC可以看作是具有受约束损失函数的投影梯度下降方法。受约束损失函数及其梯度的Lipschitzness更好，
        # 因此训练过程变得更加有效和稳定。 
        
        g_tmp = param_and_grad[1]
        if self.use_gc and g_tmp.dim() > self.gc_gradient_threshold:
            #print("grad before gc:", g_tmp)
            g_tmp = g_tmp - g_tmp.mean(axis=tuple(range(1, g_tmp.dim())), keepdim=True)
            #print("grad after gc:",g_tmp)
        
        """
        moment = self._get_accumulator(self._moment1_acc_str, param_and_grad[0])
        inf_norm = self._get_accumulator(self._inf_norm_acc_str,
                                         param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        
        # create the adamax optimize op
        adamax_op = block.append_op(
            type="adamax",
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment": moment,
                "InfNorm": inf_norm,
                "Beta1Pow": beta1_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": moment,
                "InfNormOut": inf_norm
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon
            },
            stop_gradient=True)"""
        
        # RAdam
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])
        
        block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": g_tmp,
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment1": moment1,
                "Moment2": moment2,
                "Beta1Pow": beta1_pow_acc,
                "Beta2Pow": beta2_pow_acc,
                #"StepSize": [self._step_size_var],
                #"SmaFlag": [self._sma_flag_var]
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "Moment1Out": moment1,
                "Moment2Out": moment2,
                "Beta1PowOut": beta1_pow_acc,
                "Beta2PowOut": beta2_pow_acc
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon,
                "weight_decay": self._weight_decay,
                "step_size": self._step_size,
                "sma_flag": self._sma_flag
            },
            stop_gradient=True)
        
        #print("after radam, param:", param_and_grad[0])
        #print("after radam, grad:", param_and_grad[1])

        # Look ahead
        """
        one_var = paddle.ones(shape=[1], dtype='int32', name='lookahead_ones')
        zero_var = paddle.zeros(shape=[1], dtype='int32', name='lookhead_zeros')
        k_var = layers.create_global_var(
            name=unique_name.generate("lookahead_k"),
            shape=[1],
            value=self.k,
            dtype='int32',
            persistable=True
        )
        
        # paddle.mod代替? https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/mod_cn.html#mod
        mod = paddle.mod(self._global_step_var, k_var)

        cond_1 = paddle.equal(self._global_step_var, one_var) # global step是不是等于1
        cond_1 = paddle.cast(cond_1, dtype='float32')

        cond_2 = paddle.equal(mod, zero_var) # global step%k是不是等于0
        cond_2 = paddle.cast(cond_2, dtype='float32')

        slow_var = self._get_accumulator(self._slow_str, param_and_grad[0]) # 缓存的slow var

        # 初始化slow_var
        tmp_var = cond_1 * param_and_grad[0] + (1 - cond_1) * slow_var
        paddle.assign(tmp_var, slow_var)

        # 融合model param
        tmp_var = self.alpha * param_and_grad[0] + (1.0 - self.alpha) * slow_var
        tmp_var_1 = cond_2 * tmp_var + (1 - cond_2) * param_and_grad[0]
        paddle.assign(tmp_var_1, param_and_grad[0])

        tmp_var_1 = cond_2 * tmp_var + (1 - cond_2) * slow_var
        paddle.assign(tmp_var_1, slow_var)
        """

        # look ahead的if写法
        mod = self._global_step % self.k
        slow_var = self._get_accumulator(self._slow_str, param_and_grad[0]) # 缓存的slow var
        if self._global_step == 1:
            paddle.assign(param_and_grad[0], slow_var)
        if mod == 0:
            tmp_var = self.alpha * param_and_grad[0] + (1.0 - self.alpha) * slow_var
            paddle.assign(tmp_var, slow_var)
            paddle.assign(tmp_var, param_and_grad[0])

        return None

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._slow_str, p)
            self._add_moments_pows(p)

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        # Look Ahead global_step+1
        #self._increment_global_var()
        self._global_step += 1

        # RAdam 计算N_sma和step_size，然后对于op，传入N_sma>N_sma_threshold的bool值和step_size
        self._cal_sma()

        if not isinstance(self._parameter_list[0], dict):
            params_grads = []
            for param in self._parameter_list:
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    grad_var = param._grad_ivar()
                    if hasattr(grad_var, "_is_sparse") and grad_var._is_sparse(
                    ) and self.regularization is not None:
                        raise RuntimeError(
                            "Ranger don't support weight_decay with sparse parameters, please set it to None."
                        )
                    params_grads.append((param, grad_var))
            #print(params_grads[0])
            #print(params_grads[1])
            optimize_ops = self._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads)
        else:
            # optimize parameters in groups
            for param_group in self._param_groups:
                params_grads = defaultdict(lambda: list())
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        grad_var = param._grad_ivar()
                        params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v
                     for k, v in param_group.items() if k != 'params'})
                #print(params_grads[0])
                #print(params_grads[1])
                self._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads)
