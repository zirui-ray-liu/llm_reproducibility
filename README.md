# ⚙️ Monkey Patching Solutions for Deterministic LLM Generation

*TL;DR: A toolkit that enables deterministic LLM inference and eliminates the training–inference mismatch in reinforcement learning.*


**Deterministic Inference across Tensor Parallel Sizes That Eliminates Training–Inference Mismatch** [[Paper](https://arxiv.org/abs/2511.17826)] [[Code](https://github.com/nanomaoli/llm_reproducibility/tree/main)][[Blog](https://festive-clam-15f.notion.site/Enabling-Large-Scale-True-on-Policy-RL-by-Bringing-Tensor-Parallelism-to-Order-2b039f5cabfa807b9770fcbe339f0f9b)]

**Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference** (NeurIPS 2025, **Oral**) [[Paper](https://arxiv.org/abs/2506.09501)] [[HF](https://huggingface.co/papers/2506.09501)] [[Code](https://github.com/nanomaoli/llm_reproducibility/tree/main/evaluation)]

## News
- [2026.05.29]: ⚡ **Faster TP-invariant matmul.** Added a CUDA warp-specialized Hopper (sm_90a) wgmma backend for `matmul_tp_persistent` (auto-selected for bf16, `K%256==0`, `N%128==0`; Triton fallback otherwise). On H100, same tree-reduction contract, **bitwise TP- & batch-invariant (0.000000% prob diff in the vLLM tp=1↔tp=4 test)**, with large matmul throughput gains over the Triton kernel:

  | shape (M,K,N) | Triton | CUDA (new) | speedup |
  |---|--:|--:|--:|
  | 1024,6144,2048 (Qwen3-1.7B down_proj) | 175 TFLOP/s | **326 TFLOP/s** | **1.86×** |
  | 2048,4096,4096 (8B o_proj) | 178 TFLOP/s | **293 TFLOP/s** | 1.65× |
  | 2048,25600,5120 (32B down_proj) | 176 TFLOP/s | **371 TFLOP/s** | **2.11×** |

  **Main idea:** the Triton kernel was load/barrier-bound (~175 TFLOP/s, tensor cores starved). The CUDA kernel is *warp-specialized* — **2 consumer warpgroups run `wgmma` while 1 producer warpgroup does `cp.async`**, coordinated by CUTLASS `PipelineAsync` mbarriers (no CTA-wide `__syncthreads`) with `setmaxnreg` register reallocation — so loads and tensor-core compute fully overlap. The fp32-group + bf16 balanced-tree reduction is unchanged, preserving determinism.
- [2026.04.30]: 🎉🎉🎉 Our paper on TBIK (Tree Based Invariant Kernels) has been accepted to ICML 2026!
- [2026.02.09]: 🚧 We add the custom tree all-reduce kernel to minimize the all-reduce latency (It requires NVLink).
- [2025.11.18]: 🗣️ A new paper has been released on [arxiv](https://arxiv.org/abs/2511.17826). In this paper, we proposed TBIK(Tree Based Invariant Kernels), which enables deterministic inference across TP sizes.
This kernel also fundamentally solves the training–inference mismatch problem in reinforcement learning when they are using different parallelization stragey.
- [2025.09.25]: 🎉🎉🎉 Our paper has been selected for Oral Presentation for Neurips 2025. See you in SD! 
- [2025.06.18]: Our paper has been released on [arxiv](https://arxiv.org/abs/2506.09501). Feel free to ⭐UPVOTE in [huggingface](https://huggingface.co/papers/2506.09501)

## Overview

In our [LLM evaluation reproducibility report](https://arxiv.org/pdf/2506.09501), we found that chaning Batch size and GPU counts can impact the reasoning trace a lot This is due to the differences in float-point arthmetic order across configurations. Thinking Machines Lab introduced [batch-invariant operators](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), which ensure deterministic outputs across different batch sizes. However, nondeterminism remains unresolved when changing TP sizes, a common scenario in RL that training and rollout engines employ different parallelization strategy.

<p align="center">
  <img src="figures/kernel.png" width="400"/>
</p>

<p align="center">
  <i>Illustration of TP-invariant matrix multiplication.</i>
</p>

We propose TBIK, a TP-invariant matmul method that achieves determinism by strictly controlling the reduction order in matrix multiplications. By replacing TP-size-sensitive(Row-Split) layers such as o_proj and down_proj with our TP-invariant counterparts, and by employing a tree-structured cross-GPU all-reduce, we achieve fully deterministic LLM inference across different TP sizes.

Furthermore, we align training engine FSDP (TP=1) with the vLLM rollout engine under tensor parallelism (TP>1), enabling bitwise-identical true on-policy reinforcement learning.

<p align="center">
  <img src="figures/overview.png" width="700"/>
</p>

<p align="center">
  <i>Illustration of global reductions within and across GPUs for Row-Split layers in Transformer Models.</i>
</p>


## Setup

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and setup
```bash
git clone https://github.com/nanomaoli/llm_reproducibility
cd llm_reproducibility
uv venv --python 3.12
source .venv/bin/activate
```

### 3. Install core dependencies
```bash
uv pip install -e .
```

### 4. Install mini_allreduce (custom all-reduce kernels)
```bash
uv pip install --no-build-isolation -v ./mini_allreduce
```

### 5. CUDA warp-specialized matmul backend (Hopper / sm_90)
`matmul_tp_persistent` uses a CUDA `wgmma` kernel (`src/tbik/csrc/matmul_tp.cu`) on supported inputs
(bf16, `K % 256 == 0`, `N % 128 == 0`), falling back to the Triton kernel otherwise. **No separate
pip install is needed** — it is JIT-compiled on first use (via `torch.utils.cpp_extension`, cached
under `src/tbik/csrc/.cuda_build/`), and the CUTLASS/cute headers it needs are vendored in
`src/tbik/csrc/cutlass_include/`.

Requirements for the CUDA backend:
- an **NVIDIA Hopper GPU (sm_90 / sm_90a)**, e.g. H100/H200;
- the **CUDA toolkit (`nvcc`) on `PATH`**, with a major version matching your PyTorch build
  (this project installs torch `cu128`, so use CUDA **12.x**). Check with `nvcc --version`.

The first call triggers a one-time build (~a few minutes); subsequent runs reuse the cache. If `nvcc`
is unavailable or the GPU is not Hopper, it transparently falls back to the Triton kernel.
```bash
# (optional) trigger + verify the JIT build now — should print "Is TP-Invariant: True"
TP_INVARIANT_MATMUL=1 python simple_matmul.py
```

### 6. (Optional) Install Flash Attention and TorchTitan for RL experiments
```bash
uv pip install flash-attn --no-build-isolation
git clone https://github.com/xh-ding/torchtitan.git
cd torchtitan
uv pip install -e .
cd ..
uv pip install -e ".[rl]"
```

### 7. Verify installation
```bash
python verify_installation.py
python simple_matmul.py
```

## How to use TBIK?
Unlock deterministic vLLM inference and true policy reinforcement learning with just a single, lightweight `apply_patches()` function! It automatically applies the appropriate patches based on the environment variables.

### Deterministic Matmul
Matmul operations have different reduction order when splitting over different numbers of GPUs. With our tree-based matmul, the result of matrix multiplication remains invariant regardless of the number of splits. You can switch between the standard matmul and the tree-based matmul by setting `TP_INVARIANT_MATMUL=0/1`.
```bash
TP_INVARIANT_MATMUL=1 python simple_matmul.py
```

### Deterministic LLM inference
In vLLM’s model implementation, the `o_proj` layer in the attention module and the `down_proj` layer in the FFN use 'row-split' linear layers. By replacing these row-split linear layers with our TP-invariant counterparts and co-designing both the intra-kernel reduction order and the inter-GPU reduction order(and also batch invariant operations), vLLM can achieve deterministic inference across different runtime settings.
```bash
VLLM_BATCH_INVARIANT=1 VLLM_TP_INVARIANT=1 python simple_inference.py
```

### Bitwise consistent on-policy RL
Following [spirl](https://github.com/teja-rao/spirl), we use vLLM for inference and TorchTitan for training in our demo code to make the workflow easier to try for users. In practical RL training pipelines, the training engine typically runs with FSDP (TP = 1), while the inference engine runs with TP > 1. This mismatch in numerical precision across the two engines is the fundamental cause of training instability in RL. By introducing TBIK and aligning the operators used in both the training and inference engines, we address this issue at its root, making true on-policy RL feasible.
```bash
[CUDA_VISIBLE_DEVICES] VLLM_BATCH_INVARIANT=1 VLLM_TP_INVARIANT=1 ALIGN_TRAIN_INFERENCE=1 python simple_rl.py
```

**We now also support running the training engine on multiple GPUs, enabling larger-scale RL training.** To launch the FSDP-based multi-GPU RL training setup with TBIK, run:

```bash
bash fsdp_rl_tbik.sh
```

To compare against other settings, such as the baseline or BIO, run:
```bash
bash fsdp_rl_baseline.sh
bash fsdp_rl_bio.sh
```
## Evaluating Reproducibility of Reasoning
To run the evaluation on the full dataset, please refer to [./evaluation](https://github.com/nanomaoli/llm_reproducibility/tree/main/evaluation)

## Contributing
The code was currently tested with ✅Qwen3-1.7B, should work with other Qwen3 models with the same architecture.

We welcome contributions from the research community to improve TBIK. If you have any idea or would like to report a bug, please open an issue or submit a pull request.


## Citation

If you find our work helpful, please kindly cite our paper.

```bibtex

@misc{yuan2025fp32deathchallengessolutions,
      title={Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning}, 
      author={Jiayi Yuan and Hao Li and Xinheng Ding and Wenya Xie and Yu-Jhe Li and Wentian Zhao and Kun Wan and Jing Shi and Xia Hu and Zirui Liu},
      year={2025},
      eprint={2506.09501},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.09501}, 
}

@misc{zhang2025deterministicinferencetensorparallel,
      title={Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch}, 
      author={Ziyang Zhang and Xinheng Ding and Jiayi Yuan and Rixin Liu and Huizi Mao and Jiarong Xing and Zirui Liu},
      year={2025},
      eprint={2511.17826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.17826}, 
}
```

## Acknowledgment
Our implementation of `simple_rl.py` and `torchtitan` is adapted from [spirl](https://github.com/teja-rao/spirl) repository.
