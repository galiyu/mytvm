/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cuDNN kernel calls for the forward algorithm.
 */
#include "/root/wmma/src/conv2d_sub/kernels.cuh"
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

void ConvolutionForward(int mode, int format, int algo, int dims, int groups, const int pad[],
                        const int stride[], const int dilation[], DLTensor* x, DLTensor* w,
                        DLTensor* y, const std::string& conv_dtype) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  SetConvDescriptors(entry_ptr, format, dims, groups, pad, stride, dilation, x->shape, w->shape,
                     y->shape, x->dtype, conv_dtype);
  // Set Device
  entry_ptr->conv_entry.device = x->device;
  // Set Algo
  entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);

  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      entry_ptr->conv_entry.fwd_algo, &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);
  CUDNN_CALL(cudnnConvolutionForward(
      entry_ptr->handle, CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.input_desc, x->data, entry_ptr->conv_entry.filter_desc, w->data,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.fwd_algo,
      entry_ptr->conv_entry.workspace, workspace_size,
      CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.output_desc, y->data));
}

void ConvolutionBiasActivationForward(int mode, int format, int algo, int dims, int groups, int act,
                                      double coef, const int pad[], const int stride[],
                                      const int dilation[], DLTensor* x, DLTensor* w, DLTensor* y,
                                      DLTensor* bias, const std::string& conv_dtype) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  CUDNN_CALL(cudnnSetActivationDescriptor(entry_ptr->conv_entry.activation_desc,
                                          static_cast<cudnnActivationMode_t>(act),
                                          cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN, coef));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
      entry_ptr->conv_entry.bias_desc, entry_ptr->conv_entry.tensor_format,
      CuDNNDataType::DLTypeToCuDNNType(bias->dtype), 1, static_cast<int>(w->shape[0]), 1, 1));

  SetConvDescriptors(entry_ptr, format, dims, groups, pad, stride, dilation, x->shape, w->shape,
                     y->shape, x->dtype, conv_dtype);
  // Set Device
  entry_ptr->conv_entry.device = x->device;
  // Set Algo
  entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);

  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      entry_ptr->conv_entry.fwd_algo, &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);
  CUDNN_CALL(cudnnConvolutionBiasActivationForward(
      entry_ptr->handle, CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.input_desc, x->data, entry_ptr->conv_entry.filter_desc, w->data,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.fwd_algo,
      entry_ptr->conv_entry.workspace, workspace_size,
      CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
      entry_ptr->conv_entry.output_desc, y->data, entry_ptr->conv_entry.bias_desc, bias->data,
      entry_ptr->conv_entry.activation_desc, entry_ptr->conv_entry.output_desc, y->data));
}

void FindAlgo(int format, int dims, int groups, const int pad[], const int stride[],
              const int dilation[], const int x_dim[], const int w_dim[], const int y_dim[],
              const std::string& data_dtype, const std::string& conv_dtype, TVMRetValue* ret) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  const int full_dims = dims + 2;
  std::vector<int64_t> x_dim_int64(full_dims);
  std::vector<int64_t> w_dim_int64(full_dims);
  std::vector<int64_t> y_dim_int64(full_dims);
  for (int i = 0; i < full_dims; ++i) {
    x_dim_int64[i] = x_dim[i];
    w_dim_int64[i] = w_dim[i];
    y_dim_int64[i] = y_dim[i];
  }
  SetConvDescriptors(entry_ptr, format, dims, groups, pad, stride, dilation, x_dim_int64.data(),
                     w_dim_int64.data(), y_dim_int64.data(), String2DLDataType(data_dtype),
                     conv_dtype);

  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, entry_ptr->conv_entry.filter_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc,
      CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned_algo_count, perf_results));

  const std::vector<std::string> fwd_algo_names{"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
                                                "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"};

  auto best_algo = perf_results[0].algo;
  LOG(INFO) << "\tCUDNN Found " << returned_algo_count << " fwd algorithms, choosing "
            << fwd_algo_names[best_algo];
  for (int i = 0; i < returned_algo_count; ++i) {
    LOG(INFO) << "\t\t" << i << ") " << fwd_algo_names[perf_results[i].algo]
              << " - time: " << perf_results[i].time << " ms"
              << ", Memory: " << perf_results[i].memory;
  }

  ret[0] = best_algo;
}

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.device").set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* f = args[0];
  DLTensor* k = args[1];
  DLTensor* o = args[2];
  auto feature = reinterpret_cast<void*>(static_cast<char*>(f->data) + f->byte_offset);
  auto kernel = reinterpret_cast<void*>(static_cast<char*>(k->data) + k->byte_offset);
  auto out = reinterpret_cast<void*>(static_cast<char*>(o->data) + o->byte_offset);
  int batch = args[3];
  int in_c = args[4];
  int out_channel = args[5];
  int f_w = args[6];
  int f_h = args[7];
  int k_w = args[8];
  int k_h = args[9];
  int stride = args[10];
  int padding = args[11];

  int data_n = batch, data_c = in_c, data_h = f_h, data_w = f_w;
  int kernel_n = out_channel, kernel_c = in_c, kernel_h = k_h, kernel_w = k_w;
  int dilation = 1;

  //handle
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle))

  // input
  cudnnTensorDescriptor_t input_descriptor;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor))
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                          data_n, data_c, data_h, data_w)) // n, c, h, w

  // kernel
  //printTensor(kernel, kernel_n, kernel_c, kernel_w, kernel_h, "kernel: ");
  cudnnFilterDescriptor_t kernel_descriptor;
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor))
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                                          kernel_n, kernel_c, kernel_h, kernel_w))

  // convolution descriptor
  cudnnConvolutionDescriptor_t conv_descriptor;
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_descriptor))
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                              padding, padding, // zero-padding
                                              stride, stride, // stride
                                              dilation, dilation, // dilation 卷积核膨胀 膨胀后用0填充空位
          // 卷积是需要将卷积核旋转180°再进行后续的 -> CUDNN_CONVOLUTION
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))

  // output
  int out_n, out_c, out_h, out_w;
  CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_descriptor, input_descriptor, kernel_descriptor,
                                                    &out_n, &out_c, &out_h, &out_w))

  cudnnTensorDescriptor_t output_descriptor;
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_descriptor))
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                          out_n, out_c, out_h, out_w))

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT; // no support
  //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;

  // workspace size && allocate memory
  size_t workspace_size = 0;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                      input_descriptor,
                                                      kernel_descriptor,
                                                      conv_descriptor,
                                                      output_descriptor,
                                                      algo,
                                                      &workspace_size))

  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, workspace_size))

  // convolution
  auto alpha = 1.0f, beta = 0.0f;

  // calculate
  CHECK_CUDNN(cudnnConvolutionForward(handle,
                                      &alpha, input_descriptor, feature,
                                      kernel_descriptor, kernel,
                                      conv_descriptor, algo,
                                      workspace, workspace_size,
                                      &beta, output_descriptor, out))

  // destroy
  CHECK_CUDA(cudaFree(workspace))

  CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor))
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_descriptor))
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_descriptor))
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor))

  CHECK_CUDNN(cudnnDestroy(handle))
});

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int mode = args[0];
      int format = args[1];
      int algo = args[2];
      int pad_v[2], stride_v[2], dilation_v[2];
      for (int i = 0; i < 2; i++) {
        pad_v[i] = args[3 + i];
        stride_v[i] = args[5 + i];
        dilation_v[i] = args[7 + i];
      }
      DLTensor* x = args[9];
      DLTensor* w = args[10];
      DLTensor* y = args[11];
      std::string conv_dtype = args[12];
      int groups = args[13];

      ConvolutionForward(mode, format, algo, 2, groups, pad_v, stride_v, dilation_v, x, w, y,
                         conv_dtype);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d+bias+act.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int mode = args[0];
      int format = args[1];
      int algo = args[2];
      int pad_v[2], stride_v[2], dilation_v[2];
      for (int i = 0; i < 2; i++) {
        pad_v[i] = args[3 + i];
        stride_v[i] = args[5 + i];
        dilation_v[i] = args[7 + i];
      }
      int act = args[9];
      double coef = args[10];
      DLTensor* x = args[11];
      DLTensor* w = args[12];
      DLTensor* bias = args[13];
      DLTensor* y = args[14];
      std::string conv_dtype = args[15];
      int groups = args[16];

      ConvolutionBiasActivationForward(mode, format, algo, 2, groups, act, coef, pad_v, stride_v,
                                       dilation_v, x, w, y, bias, conv_dtype);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv3d.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int mode = args[0];
      int format = args[1];
      int algo = args[2];
      int pad_v[3], stride_v[3], dilation_v[3];
      for (int i = 0; i < 3; i++) {
        pad_v[i] = args[3 + i];
        stride_v[i] = args[6 + i];
        dilation_v[i] = args[9 + i];
      }
      DLTensor* x = args[12];
      DLTensor* w = args[13];
      DLTensor* y = args[14];
      std::string conv_dtype = args[15];
      int groups = args[16];

      ConvolutionForward(mode, format, algo, 3, groups, pad_v, stride_v, dilation_v, x, w, y,
                         conv_dtype);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.forward_find_algo")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      int format = args[0];
      int dims = args[1];
      int* pad = static_cast<int*>(static_cast<void*>(args[2]));
      int* stride = static_cast<int*>(static_cast<void*>(args[3]));
      int* dilation = static_cast<int*>(static_cast<void*>(args[4]));
      int* x_dim = static_cast<int*>(static_cast<void*>(args[5]));
      int* w_dim = static_cast<int*>(static_cast<void*>(args[6]));
      int* y_dim = static_cast<int*>(static_cast<void*>(args[7]));
      std::string data_dtype = args[8];
      std::string conv_dtype = args[9];
      int groups = args[10];

      FindAlgo(format, dims, groups, pad, stride, dilation, x_dim, w_dim, y_dim, data_dtype,
               conv_dtype, ret);
    });

}  // namespace contrib
}  // namespace tvm
