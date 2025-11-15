# AlexNet Deep Neural Network Compression

This project implements the three-stage deep neural network compression pipeline (Pruning, Quantization, and Huffman Coding) as described in the paper "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding" by Han et al.

The target model for compression is a pre-trained **AlexNet** model.

## The 3-Stage Pipeline

1.  **Stage 1: Pruning**
    Weights with a low L1 magnitude (below a certain percentile threshold) are considered unimportant and are removed from the network, setting them to zero. This creates a sparse model.

2.  **Stage 2: Trained Quantization**
    The remaining non-zero weights are clustered using K-Means. All weights belonging to a cluster are replaced by the cluster's centroid value. This reduces the number of unique weight values. We store the small (e.g., 5-bit) cluster indices and a codebook of the (e.g., 32) centroid values.

3.  **Stage 3: Huffman Coding**
    The quantized cluster indices are further compressed using Huffman coding, a lossless compression algorithm that assigns shorter codes to more frequent indices.

## How to Run

### Dependencies

You will need the following Python libraries:

* `torch`
* `torchvision`
* `numpy`
* `scikit-learn` (for KMeans)

You can install them using pip:

```sh
pip install torch torchvision numpy scikit-learn
python deep_nn_compression.py

Loading pre-trained AlexNet...

--- Model Size Info ---
Total parameters: 57,044,810
Original model size (32-bit floats): 228.18 MB
------------------------

--- STAGE 1: PRUNING (Target 90.0%) ---
Pruning threshold: 0.01783584
Total non-zero weights: 5,703,547 / 57,035,456
Achieved sparsity: 90.00%

--- Model Size Info ---
Total parameters: 57,044,810
Original model size (32-bit floats): 228.18 MB
Non-zero weights after pruning: 5,703,547
Sparsity: 90.00%
Pruned model size (still 32-bit): 22.81 MB
------------------------

--- STAGE 2: QUANTIZATION (32 clusters) ---
Running KMeans on 5,703,547 weights...
Codebook (centroids) shape: (32,)
Labels (indices) shape: (5703547,)

--- STAGE 3: HUFFMAN CODING ---
Frequency distribution of the 32 weight indices (top 5):
[(7, 1060071),
 (10, 855284),
 (29, 780911),
 (20, 618101),
 (18, 547480)]

Huffman codes (sample):
  Index 7: 00 (Freq: 1060071)
  Index 10: 110 (Freq: 855284)
  Index 29: 101 (Freq: 780911)
  Index 20: 011 (Freq: 618101)
  Index 18: 1000 (Freq: 547480)

Compressed indices: 20,350,607 bits
Average bits per index: 3.57 bits
(Compared to 5 bits for fixed-length encoding)
Codebook size: 1,024 bits (32 * 32-bit floats)
Sparse metadata size (uncompressed): 182,513,568 bits

--- FINAL COMPRESSION SUMMARY ---
Original Model Size:       228.1792 MB
After Pruning (90%):       45.6284 MB
After Pruning + Quant (Fixed 5-bit): 26.3790 MB
After Pruning + Quant + Huffman: 25.3581 MB

Compression Ratios (vs. Original):
  Pruning only:            5.00x
  Pruning + Quant (Fixed): 8.65x
  Full Pipeline (Huffman): 9.00x
