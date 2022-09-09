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

@tvm.register_func("tvm.contrib.my_tvm_matmul_temp")
def my_tvm_matmul_temp(a, b, c, m, k, n):
    _Lib = ctypes.CDLL("/root/wmma/lib_wmma/libwmmaapi.so", ctypes.RTLD_GLOBAL)

    # A_data = a.numpy().flatten('c').astype(c_double)
    # B_data = b.numpy().flatten('f').astype(c_double)
    A_data = np.ones(m*k).flatten().astype(c_double)
    B_data = np.ones(k*n).flatten().astype(c_double)
    C_data = np.random.uniform(-1, 1, m*n).astype(c_float)
    A = (ctypes.c_double*len(A_data))(*A_data)
    B = (ctypes.c_double*len(B_data))(*B_data)
    C = (ctypes.c_float*len(C_data))(*C_data)
    _Lib.wmma_api(A, B, C, m, n, k)
    # print(np.array(C))
    tvm.nd.array( np.array(C).astype("float64").reshape(m,n) ).copyto(c)


@tvm.register_func("tvm.contrib.im2col_temp")
def im2col_temp(data_im, data_col, data_n, channels,
                height, width, kernel_h, kernel_w, out_w, out_h,
                pad_h, pad_w,
                stride_h, stride_w,
                dilation_h = 1, dilation_w = 1):
    _Lib = ctypes.CDLL("/root/wmma/lib_conv2d_sub/lib_im2col/libim2col.so", ctypes.RTLD_GLOBAL)

    im = data_im.numpy().flatten().astype(c_float)
    col = data_col.numpy().flatten().astype(c_float)
    im_Z = (ctypes.c_float*len(im))(*im)
    col_Z = (ctypes.c_float*len(col))(*col)
    
    _Lib.im2col_gpu(im_Z, data_n, channels,
                height, width, kernel_h, kernel_w,
                pad_h, pad_w,
                stride_h, stride_w,
                dilation_h, dilation_w, col_Z)
    # print("======im2col_out=====")
    # print(np.array(col_Z))
    tvm.nd.array( np.array(col_Z).astype("float32").reshape(data_n, out_w, out_h, channels, kernel_w, kernel_h) ).copyto(data_col)

@tvm.register_func("tvm.contrib.gemm_temp")
def gemm_temp(im2col_out, kernel, gemm_out,
                M, K, N,
                ):
    _Lib = ctypes.CDLL("/root/wmma/lib_conv2d_sub/lib_gemm_cublas/libgemmcublas.so", ctypes.RTLD_GLOBAL)
    outdata = gemm_out.numpy()
    outshape = outdata.shape
    
    input_A = im2col_out.numpy().flatten().astype(c_float)
    input_B = kernel.numpy().flatten().astype(c_float)
    output = outdata.flatten().astype(c_float)
    input_A_Z = (ctypes.c_float*len(input_A))(*input_A)
    input_B_Z = (ctypes.c_float*len(input_B))(*input_B)
    output_Z = (ctypes.c_float*len(output))(*output)

    _Lib.cublas_gemm_host(input_B_Z, input_A_Z, M, K, N, output_Z)
    # print("======gemm_out======")
    # print(np.array(output_Z))
    tvm.nd.array( np.array(output_Z).astype("float32").reshape(outshape)).copyto(gemm_out)

@tvm.register_func("tvm.contrib.col2im_here")
def col2im_temp(data, col2im_out, data_n, kernel_n, out_h, out_w):
    _Lib = ctypes.CDLL("/root/wmma/lib_conv2d_sub/lib_col2im/libcol2im.so", ctypes.RTLD_GLOBAL)
    out_data = col2im_out.numpy()
    out_shape = out_data.shape
    input = data.numpy().flatten().astype(c_float)

    output = out_data.flatten().astype(c_float)
    input_Z = (ctypes.c_float*len(input))(*input)
    output_Z = (ctypes.c_float*len(output))(*output)

    _Lib.col2im_gpu(input_Z, data_n, kernel_n, out_h, out_w, output_Z)
    # print("======col2im_out======")
    # print(np.array(output_Z))
    tvm.nd.array( np.array(output_Z).astype("float32").reshape(out_shape) ).copyto(col2im_out)

def compute_my_im2col(
    data, kernel, strides, padding, dilation=1, groups=1, layout="NCHW", out_dtype="float32"
):
    batch, in_channel, f_h, f_w = get_const_tuple(data.shape)
    out_channel, _, k_h, k_w = get_const_tuple(kernel.shape)
    stride_h, stride_w = (strides, strides) if isinstance(strides, int) else strides
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation
    KH_dilated = (k_h - 1) * dilation_h + 1
    KW_dilated = (k_w - 1) * dilation_h + 1

    pt, pl, pb, pr = get_pad_tuple(padding, (KH_dilated, KW_dilated))
    padding_h = pt
    padding_w = pl

    out_w = (f_w + 2 * padding_w - k_w) // stride_w + 1
    out_h = (f_h + 2 * padding_h - k_h) // stride_h + 1

    out = te.extern(
        (batch, out_w, out_h, in_channel, k_w, k_h),
        [data],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.im2col_temp", ins[0], outs[0],
        batch, in_channel, f_w, f_h, k_w, k_h, out_w, out_h, padding_h, padding_w, stride_h, stride_w
        ),
        name="compute_my_im2col",
    )

    return out

def schedule_my_im2col(outs):
    """Create the schedule for conv2d"""
    return generic.schedule_extern(outs)

def compute_my_gemm(
    data, kernel, strides, padding, dilation, groups=1, layout="NCHW", out_dtype="float32"
):
    batch, out_w, out_h, in_channel, k_w, k_h= get_const_tuple(data.shape)
    out_channel, _, k_h, k_w = get_const_tuple(kernel.shape)

    M = out_channel
    K = in_channel*k_h*k_w
    N = batch*out_w*out_h
    
    out = te.extern(
        (batch, out_channel, out_w, out_h),
        [data, kernel],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.gemm_temp", ins[0], ins[1], outs[0],
        M, K, N
        ),
        name="compute_my_gemm",
    )
    return out

def schedule_my_gemm(outs):
    """Create the schedule for conv2d"""
    return generic.schedule_extern(outs)

def compute_my_col2im(
    data, kernel, strides, padding, dilation, groups=1, layout="NCHW", out_dtype="float32"
):
    batch, out_channel, out_h, out_w= get_const_tuple(data.shape)
    
    out = te.extern(
        (batch, out_channel, out_w, out_h),
        [data],
        lambda ins, outs: tvm.tir.call_packed("tvm.contrib.col2im_here", ins[0], outs[0],
        batch, out_channel, out_h, out_w
        ),
        name="compute_my_col2im",
    )
    return out

def schedule_my_col2im(outs):
    """Create the schedule for conv2d"""
    return generic.schedule_extern(outs)

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
