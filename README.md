# MicroSwag Project

## Project Overview
MicroSwag is a project focused on optimizing small language models (124M base models) for performance on the HellaSwag benchmark. The project explores the architectural efficiency frontier for compact language models through a systematic comparison of modern transformer architectures, all while maintaining strict parameter and training constraints.

**It also is a codebase used to pre-train LLMs. It is very simplistic and meant for smaller experiments. It squeezes out a lot of performance that you can get through vanilla pytorch. Being designed for small models, a lot of big model optimizations are missing.**

## Research Objectives
- Determine which transformer architectures perform best at small parameter scales (< 200M parameters)

## Methodology
- Pretrain multiple model architectures on the FineWeb 10B dataset for exactly one epoch, batch size 0.5M tokens, (19073 steps)
- Maintain consistent training configuration across experiments (data loading, optimization, etc.), will change training methods if explitly mentioned in paper
- Implement and evaluate 4-5 different model architectures including:
  - GPT-2 (baseline, karpathy vid)
  - LLaMA 3
  - Phi-4
  - Gemma 3
  - Mistral
  - DeepSeek-V3 (MoE)
  - RWKV
- Measure and compare HellaSwag performance across architectures

## Constraints
- Parameter budget: <= 124M parameter base model (may move higher post-training)
- Training data: FineWeb 10B dataset only (single epoch)
- Training infrastructure: 8xA100 80GB GPUs (most likely)
- Training time target: under 2 hours per model architecture

## Research Questions
- How do dense transformers compare to sparse MoE architectures at small scales?
- Do architectural innovations from larger models (like rotary positional embeddings, normalization strategies) transfer effectively to smaller scales?
- Which attention mechanisms work best for models in this parameter range?
- How do activation functions impact model performance at this scale?
- Can RNN architectures (RWKV) compete?

## Future Directions
- Post-training optimizations for the best-performing architecture
- Exploration of hybrid architectures combining the best elements from multiple designs
- Investigation of task-specific architectural modifications for HellaSwag performance
- Scaling insights that might apply to both smaller and larger model regimes
