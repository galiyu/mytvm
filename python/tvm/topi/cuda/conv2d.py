# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""Compute definition for conv2d with cuda backend"""
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import OtherOptionEntity
from tvm.contrib import cudnn

from .. import nn, generic
from ..nn.utils import get_pad_tuple
from ..utils import get_const_tuple, traverse_inline
from .conv2d_direct import schedule_direct_cuda


@autotvm.register_topi_compute("conv2d_nchw.cuda")
def conv2d_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d with NCHW layout"""
    return nn.conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nchw.cuda")
def schedule_conv2d_nchw(cfg, outs):
    """Create the schedule for conv2d_nchw"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv2d_nchw":
            schedule_direct_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s

import tvm
from tvm import te,topi
import numpy as np
import ctypes
from ctypes import *
@tvm.register_func("tvm.contrib.my_tvm_im2col_temp")
def my_tvm_im2col_temp(a, b, p_N, p_C, p_H, p_W, p_K, p_P, p_Q, p_KH, p_KW, p_SH, p_SW, p_L, p_R, p_T, p_B):
    _Lib = ctypes.CDLL("/root/wmma/lib_im2col/libim2col.so", ctypes.RTLD_GLOBAL)

    data_A = a.numpy().flatten().astype(c_float)
    data_B = np.zeros(p_C*p_KH*p_KW*p_N*p_P*p_Q).astype(c_float)
    data_A_Z = (ctypes.c_float*len(data_A))(*data_A)
    data_B_Z = (ctypes.c_float*len(data_B))(*data_B)

    _Lib.im2col_api(data_A_Z, data_B_Z, p_N, p_C, p_H, p_W, p_K, p_P, p_Q, p_KH, p_KW, p_SH, p_SW, p_L, p_R, p_T, p_B )
    return tvm.nd.array( np.array(data_B_Z).astype("float32").reshape(p_C, p_KH, p_KW, p_N, p_P, p_Q) )

@tvm.register_func("tvm.contrib.my_tvm_gemm_temp")
def my_tvm_gemm_temp(a, b, c, p_N, p_C,p_K, p_P, p_Q, p_KH, p_KW, p_SH, p_SW, p_L, p_R, p_T, p_B ):
    _Lib = ctypes.CDLL("/root/wmma/lib_gemm_cublas/libgemmcublas.so", ctypes.RTLD_GLOBAL)
    
    data_A_im2col = a.numpy().flatten().astype(c_float)
    kernel_B = b.numpy().flatten().astype(c_float)
    out_C = np.random.uniform(-1,1, p_N*p_K*p_P*p_Q).astype(c_float)
    
    data_A_im2col_Z = (ctypes.c_float*len(data_A_im2col))(*data_A_im2col)
    kernel_B_Z = (ctypes.c_float*len(kernel_B))(*kernel_B)
    out_C_Z = (ctypes.c_float*len(out_C))(*out_C)
    
    _Lib.gemm_cublas_api(data_A_im2col_Z, kernel_B_Z, out_C_Z, p_N, p_C, p_K, p_P, p_Q, p_KH, p_KW, p_SH, p_SW, p_L, p_R, p_T, p_B );
    return tvm.nd.array( np.array(out_C_Z).astype("float32").reshape(p_N, p_K, p_P, p_Q) )

@tvm.register_func("tvm.contrib.my_tvm_col2im_temp")
def my_tvm_col2im_temp(a, b, p_N, p_C,p_K, p_P, p_Q, p_KH, p_KW, p_SH, p_SW, p_L, p_R, p_T, p_B):
    _Lib = ctypes.CDLL("/root/wmma/lib_col2im/libcol2im.so", ctypes.RTLD_GLOBAL)
    
    out_C = a.numpy().flatten().astype(c_float)
    out_C_col2im = np.zeros(p_N*p_P*p_Q*p_K).astype(c_float)  
    out_C_Z = (ctypes.c_float*len(out_C))(*out_C)
    out_C_col2im_Z = (ctypes.c_float*len(out_C_col2im))(*out_C_col2im)
    
    _Lib.col2im_api(out_C_Z, out_C_col2im_Z, p_N, p_C, p_K, p_P, p_Q, p_KH, p_KW, p_SH, p_SW, p_L, p_R, p_T, p_B )
    return tvm.nd.array( np.array(out_C_col2im_Z).astype("float32").reshape(p_N, p_K, p_P, p_Q) )

@tvm.register_func("tvm.contrib.my_tvm_matmul_temp")
def my_tvm_matmul_temp(a, b, c, m, k, n):
    _Lib = ctypes.CDLL("/root/wmma/lib_wmma/libwmmaapi.so", ctypes.RTLD_GLOBAL)

    A_data = a.numpy().flatten('c').astype(c_double)
    B_data = b.numpy().flatten('f').astype(c_double)
    C_data = np.random.uniform(-1, 1, m*n).astype(c_float)
    A = (ctypes.c_double*len(A_data))(*A_data)
    B = (ctypes.c_double*len(B_data))(*B_data)
    C = (ctypes.c_float*len(C_data))(*C_data)
    _Lib.wmma_api(A, B, C, m, n, k)
    tvm.nd.array( np.array(C).astype("float64").reshape(m,n) ).copyto(c)

def compute_my_im2col(
    data, kernel, strides, padding, dilation, groups=1, layout="NHWC", out_dtype="float32"
):
    N, C, H, W = get_const_tuple(data.shape)
    print((N, C, H, W))
    K, _, KH, KW = get_const_tuple(kernel.shape)
    
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    KH_dilated = (KH - 1) * dilation_h + 1
    KW_dilated = (KW - 1) * dilation_h + 1
    SH = stride_h
    SW = stride_w

    pt, pl, pb, pr = get_pad_tuple(padding, (KH_dilated, KW_dilated))
    
    P = (H + pt + pb - KH) // stride_h + 1
    Q = (W + pl + pr - KW) // stride_w + 1
    
    out = te.extern(
        [C, KH, KW, N, P, Q],   # CHWNPQ
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.my_tvm_im2col_temp", ins[0], outs[0],
            N, C, H, W, K, P, Q, KH, KW, SH, SW, pl, pr, pt, pb
        ),
        name="compute_my_im2col",
    )
    return out

def compute_my_gemm(
    data, kernel, strides, padding, dilation, groups=1, layout="NHWC", out_dtype="float32"
):
    C, KH, KW, N, P, Q= get_const_tuple(data.shape)
    print((C, KH, KW, N, P, Q))
    K, _, KH, KW = get_const_tuple(kernel.shape)
    
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    KH_dilated = (KH - 1) * dilation_h + 1
    KW_dilated = (KW - 1) * dilation_h + 1
    SH = stride_h
    SW = stride_w

    pt, pl, pb, pr = get_pad_tuple(padding, (KH_dilated, KW_dilated))
    
    out = te.extern(
        [N, K, P, Q],   # CHWNPQ
        [data, kernel],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.my_tvm_gemm_temp", ins[0], ins[1], outs[0],
            N, C, K, P, Q, KH, KW, SH, SW, pl, pr, pt, pb
        ),
        name="compute_my_gemm",
    )
    return out

def compute_my_col2im(
    data, kernel, strides, padding, dilation, groups=1, layout="NHWC", out_dtype="float32"
):
    N, K, P, Q= get_const_tuple(data.shape)
    print((N, K, P, Q))
    K, C, KH, KW = get_const_tuple(kernel.shape)
    
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    KH_dilated = (KH - 1) * dilation_h + 1
    KW_dilated = (KW - 1) * dilation_h + 1
    SH = stride_h
    SW = stride_w

    pt, pl, pb, pr = get_pad_tuple(padding, (KH_dilated, KW_dilated))
    
    out = te.extern(
        [N, K, P, Q],   # CHWNPQ
        [data, kernel],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.my_tvm_col2im_temp", ins[0], outs[0],
            N, C, K, P, Q, KH, KW, SH, SW, pl, pr, pt, pb
        ),
        name="compute_my_col2im",
    )
    return out

def compute_my_matmul(A, B, units=None, out_dtype="", transpose_a=False, transpose_b=False):
    m, k = get_const_tuple(A.shape)
    _, n = get_const_tuple(B.shape)
    
    
    out = te.extern(
        [m, n],
        [A, B],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.my_tvm_matmul_temp", ins[0], ins[1], outs[0],
            m, k, n
        ),
        name = "compute_my_matmul",
    )
    return out

def compute_my_matadd(A, B, units=None, out_dtype="", transpose_a=False, transpose_b=False):
    out = te.compute(
        A.shape,
        lambda i,j,k,l: A[i,j,k,l] + B[i,j,k,l],
        name = "compute_my_matadd",
    )
    return out


@autotvm.register_topi_compute("conv2d_cudnn.cuda")
def conv2d_cudnn(
    cfg, data, kernel, strides, padding, dilation, groups=1, layout="NCHW", out_dtype="float32"
):
    """Compute conv2d using CuDNN library"""
    if layout == "NCHW":
        tensor_format = 0  # CUDNN_TENSOR_NCHW
        N, _, H, W = get_const_tuple(data.shape)
    elif layout == "NHWC":
        tensor_format = 1  # CUDNN_TENSOR_NHWC
        N, H, W, _ = get_const_tuple(data.shape)
    else:
        raise ValueError("Unsupported layout %s in cudnn" % layout)
    CO, CI, KH, KW = get_const_tuple(kernel.shape)

    # handle dilation
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    KH_dilated = (KH - 1) * dilation_h + 1
    KW_dilated = (KW - 1) * dilation_h + 1

    pt, pl, pb, pr = get_pad_tuple(padding, (KH_dilated, KW_dilated))
    if (pt != pb) or (pl != pr):
        raise ValueError("Cudnn doesn't support asymmetric padding.")

    OH = (H + pt + pb - KH) // stride_h + 1
    OW = (W + pl + pr - KW) // stride_w + 1

    if isinstance(N, int):
        cfg.add_flop(
            groups
            * 2
            * N
            * OH
            * OW
            * CO
            * CI
            * ((KH - 1) * dilation_h + 1)
            * ((KW - 1) * dilation_w + 1)
        )

    if data.dtype == "int8" or kernel.dtype == "int8":
        if layout == "NCHW":
            raise ValueError("NCHW layout do not support int8 in cudnn")
        dtype = "int32"
    else:
        dtype = data.dtype

    cfg.define_knob("algo", range(cudnn.algo_to_index("fwd", "CUDNN_CONVOLUTION_FWD_ALGO_COUNT")))
    if cfg.is_fallback:
        if cudnn.exists():
            # Let CUDNN choose the best algo, based on benchmarks run
            # on the local machine.  In the future, this should be
            # based on parameters stored in the Target.
            cfg["algo"] = OtherOptionEntity(-1)
        else:
            cfg["algo"] = OtherOptionEntity(0)

    return cudnn.conv_forward(
        data,
        kernel,
        [pt, pl],  # cudnn padding pt, pl on both sides of input
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        conv_mode=1,
        tensor_format=tensor_format,
        algo=cfg["algo"].val,
        conv_dtype=dtype,
        groups=groups,
    )


@autotvm.register_topi_schedule("conv2d_cudnn.cuda")
def schedule_conv2d_cudnn(cfg, outs):
    """Create the schedule for conv2d_cudnn"""
    return generic.schedule_extern(outs)


def conv2d_backward_weight_cudnn(
    dy, x, kernel_size, padding, stride, dilation, groups, layout, output_dtype
):
    """Compute conv2d wgrad using CuDNN library"""
    assert layout in ["NCHW", "NHWC"]

    if dy.dtype == "float16":
        # cuDNN does not seem to support other combination.
        assert output_dtype == "float16", "Only supports fp16 output for cuDNN fp16 wgrad."

    conv_dtype = "float32"  # Accumulation is always fp32
    return cudnn.conv_backward_filter(
        dy,
        x,
        kernel_size,
        padding,
        stride,
        dilation,
        conv_mode=1,
        tensor_format=0 if layout == "NCHW" else 1,
        conv_dtype=conv_dtype,
        groups=groups,
    )
