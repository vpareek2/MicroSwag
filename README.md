# MicroSwag Project

## Project Overview
MicroSwag is a research project focused on optimizing small language models (under 200M parameters) for performance on the HellaSwag benchmark. The project explores the architectural efficiency frontier for compact language models through a systematic comparison of modern transformer architectures, all while maintaining strict parameter and training constraints.

**It also is a codebase used to pre-train LLMs. It is very simplistic and meant for smaller experiments, not prod.**

## Research Objectives
- Determine which transformer architectures perform best at small parameter scales (< 200M parameters)
- Analyze the impact of key architectural innovations when scaled down to smaller models
- Establish performance baselines for different architectures on the HellaSwag benchmark
- Identify which architectural components contribute most significantly to performance improvements
- Create a knowledge base of architectural design principles that work well for smaller models

## Methodology
- Pretrain multiple model architectures on the FineWeb 10B dataset for exactly one epoch
- Maintain consistent training configuration across experiments (data loading, optimization, etc.)
- Implement and evaluate 4-5 different model architectures including:
  - GPT-2 (baseline, karpathy vid)
  - LLaMA 3
  - Phi-4
  - Gemma 3
  - Mistral
  - DeepSeek-V3 (MoE)
- Measure and compare HellaSwag performance across architectures

## Constraints
- Parameter budget: < 200M parameters per model
- Training data: FineWeb 10B dataset only (single epoch)
- Training infrastructure: 8xA100 80GB GPUs
- Training time target: ~2 hours per model architecture

## Research Questions
- How do dense transformers compare to sparse MoE architectures at small scales?
- Do architectural innovations from larger models (like rotary positional embeddings, normalization strategies) transfer effectively to smaller scales?
- Which attention mechanisms work best for models in this parameter range?
- How do activation functions impact model performance at this scale?

## Expected Outcomes
- A comprehensive comparison of model architectures at the < 200M parameter scale
- Insights into the scaling laws and efficiency frontiers for different architectural designs
- Identification of the most promising architectural directions for small, efficient language models
- Technical know-how for implementing and training various model architectures efficiently

## Future Directions
- Potential post-training optimizations for the best-performing architecture
- Exploration of hybrid architectures combining the best elements from multiple designs
- Investigation of task-specific architectural modifications for HellaSwag performance
- Scaling insights that might apply to both smaller and larger model regimes
