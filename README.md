# Deep-Neural-Network-Compression
This project compresses the AlexNet deep neural network using a three-stage pipeline based on the "Deep Compression" paper. It first prunes unimportant weights, then quantizes the remaining weights using k-means clustering. Finally, it applies Huffman coding to the quantized indices, achieving a 9x reduction in the model's file size.
