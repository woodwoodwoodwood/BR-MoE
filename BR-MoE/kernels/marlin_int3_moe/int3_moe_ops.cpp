/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
int brmoe_cuda_with_zeros(
  const void* A,
  const void* B1,
  const void* B2,
        void* C,
        void* s,
        void* z,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void mul_3bit_with_zeros(
  const torch::Tensor& A,
  const torch::Tensor& B1,
  const torch::Tensor& B2,
        torch::Tensor& C,
  const torch::Tensor& s,
  const torch::Tensor& z,
        torch::Tensor& workspace,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = (s.size(0) == 1) ? -1 : prob_k / s.size(0);
  if (groupsize != -1 && groupsize * s.size(0) != prob_k)
    AT_ERROR("k=", prob_k, " not compatible with ", s.size(0), " groups.");
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  int dev = A.get_device();
  int err = brmoe_cuda_with_zeros(
    A.data_ptr(),
    B1.data_ptr(),
    B2.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    z.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    groupsize,
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, ", groupsize=", groupsize, "."
    );
  }
}

int brmoe_moe_with_zeros(
  const void* A, const void* B1, const void* B2, void* C,
  const void* s, const void* z,
  int prob_n, int prob_k,
  const void* expert_ids, const void* num_post_ptr, int m_blocks_max,
  long b1_e_stride, long b2_e_stride, long s_e_stride, long z_e_stride,
  int groupsize, int k_splits, void* C32,
  int thread_n, int thread_k, int stages, int thread_m,
  int dev, cudaStream_t stream
);

// grouped MoE: A_sorted/C_sorted 都在 sorted 空间 (调用方用 index_select/index_add_
// 做 gather/scatter)。B1/B2/s/z 带专家维 [E, ...]。
void mul_3bit_moe(
  const torch::Tensor& A,         // [m_blocks_max*64, K] fp16
  const torch::Tensor& B1,        // [E, K//16, N] int32 (每专家 Marlin 布局)
  const torch::Tensor& B2,        // [E, K//16, N//2] int32
        torch::Tensor& C,         // [m_blocks_max*64, N] fp16
  const torch::Tensor& s,         // [E, K//gs, N] fp16
  const torch::Tensor& z,         // [E, K//gs, N] fp16
  const torch::Tensor& expert_ids,   // [m_blocks_max] int32 (device)
  const torch::Tensor& num_post,     // [1] int32 (device)
  int m_blocks_max,
  // split-K: k_splits>1 时部分和写 C32 (fp32, 调用方预清零), C 不写
  const torch::optional<torch::Tensor>& C32 = torch::nullopt,
  int k_splits = 1,
  int thread_n = 128, int thread_k = 128, int stages = 4, int thread_m = 16
) {
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  int groupsize = prob_k / s.size(1);
  int dev = A.get_device();
  TORCH_CHECK(thread_m == 16 || thread_m == 32 || thread_m == 64, "thread_m must be 16/32/64");
  TORCH_CHECK(thread_m == 16 || groupsize == 64, "large MoE tiles are validated only for group_size=64");
  TORCH_CHECK(A.size(0) >= (long)m_blocks_max * thread_m && C.size(0) >= (long)m_blocks_max * thread_m,
              "sorted buffers must cover m_blocks_max * thread_m rows");
  if (k_splits > 1) {
    TORCH_CHECK(C32.has_value(), "k_splits>1 需要 C32 (fp32 部分和, 预清零)");
    TORCH_CHECK(C32->scalar_type() == at::kFloat, "C32 必须是 fp32");
    TORCH_CHECK(C32->numel() >= (long)A.size(0) * prob_n,
                "C32 形状应为 [m_blocks_max*16, N]");
  }
  // stride 单位换算成 kernel 指针的元素单位: B1 是 I2(2xint32), s/z 是 int4(8xhalf)
  long b1_e = B1.stride(0) / 2;
  long b2_e = B2.stride(0);
  long s_e = s.stride(0) / 8;
  long z_e = z.stride(0) / 8;
  int err = brmoe_moe_with_zeros(
    A.data_ptr(), B1.data_ptr(), B2.data_ptr(), C.data_ptr(),
    s.data_ptr(), z.data_ptr(),
    prob_n, prob_k,
    expert_ids.data_ptr(), num_post.data_ptr(), m_blocks_max,
    b1_e, b2_e, s_e, z_e,
    groupsize, k_splits, C32.has_value() ? C32->data_ptr() : nullptr,
    thread_n, thread_k, stages, thread_m,
    dev, at::cuda::getCurrentCUDAStream(dev)
  );
  if (err == ERR_PROB_SHAPE)
    AT_ERROR("MoE problem (n=", prob_n, ", k=", prob_k, ") 形状不兼容");
  else if (err == ERR_KERN_SHAPE)
    AT_ERROR("groupsize=", groupsize, " 无对应 kernel 实例 (需要 gs/16 in {4,8})");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("supports_moe_thread_m") = true;
  m.def("mul_3bit_with_zeros", &mul_3bit_with_zeros, "BRMoE FP16xINT3 matmul with zeros.");
  m.def("mul_3bit_moe", &mul_3bit_moe, "BRMoE grouped MoE FP16xINT3 (sorted space).",
        py::arg("A"), py::arg("B1"), py::arg("B2"), py::arg("C"),
        py::arg("s"), py::arg("z"), py::arg("expert_ids"), py::arg("num_post"),
        py::arg("m_blocks_max"), py::arg("C32") = py::none(), py::arg("k_splits") = 1,
        py::arg("thread_n") = 128, py::arg("thread_k") = 128, py::arg("stages") = 4,
        py::arg("thread_m") = 16);
}
