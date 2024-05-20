# llm_int8

Contributors: Ritika Saboo, Charvi Gupta, Rhitvik Sinha 

In this project, we implement the quantization scheme outlined in the "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (https://arxiv.org/abs/2208.07339) for transformer inference. Leveraging the efficiency of CUDA, we incorporate int8 quantized matrix multiplications to optimize performance and reduce memory footprint. We implement attention, MLP and other sublayers to develop a full-fledged int8-quantized transformer model for inference. 

Quantized Transformer with Int8 Matrix Multiplication

Overview
In this project, we developed custom CUDA kernels to implement a fully quantized transformer model. Our approach leverages int8 quantized matrix multiplications, optimizing both performance and memory usage for transformer-based architectures.

Key Features
Int8 Quantized Matrix Multiplication: Implemented absmax vector-wise quantization for faster and more memory-efficient matrix operations.
Transformer Architecture: Built encoder and decoder stacks with attention mechanisms following the "Attention is all you need" framework.
Attention Layer: Developed a dedicated attention layer to support self and cross-attention during inference.
Layer Normalization: Added kernel support for residual connections and normalization between layers.
Performance Evaluation
Quantization Error: Mean quantization error is 4.58078e-05.
GEMM Timing: Unquantized GEMM: 0.31954 ms, Quantized GEMM: 1.33682 ms.
Inference Efficiency: Significant performance gains for transformer models with sizes greater than 6.7 billion parameters.

Contributions
Ritika: Vector-wise quantization, dequantize operations, timing and error analysis, layer normalization, self-attention and multi-headed attention, debugging, and encoder implementation.
Rhitvik: Attention operators, transformer architecture, and test cases for Softmax and attention layers.
Charvi: Quantization operators, tests, self-attention validation, and report preparation.

References
[High Performance Natural Language Processing] (https://aclanthology.org/2020.emnlp-tutorials.4) (Ilharco et al., EMNLP 2020) 
Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.Int8(): 8-bit Matrix Multiplication for Transformers at Scale. ArXiv. /abs/2208.07339
https://github.com/TimDettmers/bitsandbytes
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. ArXiv. /abs/1706.03762
