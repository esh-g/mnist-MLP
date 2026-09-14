# Handwritten Digit Classification using a Feedforward Neural Network (MLP) from Scratch

### 1. Network Architecture and Forward Pass (Deliverables 1.1 & 1.2)
For this project, I built a two-layer feedforward neural network from scratch using just Python and NumPy. I referred to the book "Introduction to Deep Learning" by Eugene Charniak in the BITS Library, where he recommends a $784 \to 128 \to 10$ architecture for the MNIST dataset along with cross-entropy loss. I also watched Welsh Labs' Neural Networks Demystified series and Sentdex's Neural Networks from Scratch (NNFS) videos to understand activation functions and writing neural networks without libraries.

**Network Structure**
- Input Layer ($X$): 784 neurons, corresponding to flattened $28 \times 28$ grayscale images normalized to the range $[0.0, 1.0]$.
- Hidden Layer ($z^{[1]}, a^{[1]}$): 128 neurons with ReLU activation.
- Output Layer ($z^{[2]}, a^{[2]}$): 10 neurons (digits 0 to 9) with numerically stable Softmax activation.

**Note:** There is a slight discrepency in my handwritten notes and code where in the notes I referred to input layer as 1, hidden as 2 and output as 3. Whereas in the code, weights1, biases1 correspond to hidden layer and weights2, biases2 correspond to last layer.

**Weight Initialisation**
I first initialised the weights randomly in the desired dimension, but that give me softmax probabilities that were very rigid, like they were either just 1 or just 0, So I found out it was because I hadn't used something called "He/Xavier initialization" that involves diving by sqrt of dimension to soften out the probabilities in the final step.
- $W^{[1]}$ has shape $(784, 128)$, initialized using standard normal random values scaled by $\sqrt{2 / 784}$ (He initialization for ReLU).
- $b^{[1]}$ has shape $(1, 128)$, initialized to zeros.
- $W^{[2]}$ has shape $(128, 10)$, scaled by $\sqrt{1 / 128}$
- $b^{[2]}$ has shape $(1, 10)$, initialized to zeros.


**Forward Pass Equations**

Using row vector batch notation:
1. Hidden Layer:

$$z^{[1]} = X W^{[1]} + b^{[1]}$$$$a^{[1]} = \text{ReLU}\left(z^{[1]}\right) = \max\left(0, z^{[1]}\right)$$

2. Output Layer:

$$z^{[2]} = a^{[1]} W^{[2]} + b^{[2]}$$$$a^{[2]} = \hat{y} = \text{Softmax}\left(z^{[2]}\right) = \frac{e^{z^{[2]} - \max(z^{[2]})}}{\sum e^{z^{[2]} - \max(z^{[2]})}}$$

(I shifted $z^{[2]}$ by subtracting its row-wise maximum to avoid numerical overflow with large exponentials).

3. Loss Function:
I used Categorical Cross-Entropy loss. Since the target labels are one-hot encoded, only the correct class $c$ contributes to the loss:

$$L = -\sum_{k=1}^{10} y_k \log(\hat{y}_k) = -\log(\hat{y}_c)$$

In NumPy, this was computed by indexing the predicted probabilities of the true class labels and adding $10^{-15}$ to avoid $\log(0)$ errors.

### 2. Backward Pass & Mathematical Derivations (Deliverable 1.3)
To backpropagate the error, I derived the gradients using the chain rule. It was really interesting to know how you differentiate a multivariable function with respect to an entire matrix. Welch labs videos would be the ones that give me the intuition for the matrix of derivatives. I have derived this properly in the (handwritten_notes pdf)[https://github.com/esh-g/mnist-MLP/blob/main/Backpropagation_handwritten_compressed.pdf] to prove this is not just AI slop. I do not provide the exact derivation here as it is very hard to write it out in latex and I want to use as little help from LLMs as possible, so please checkout the pdf in this repo.

### 3. Checking Gradients with PyTorch Autograd (Deliverable 1.4)
To verify that my manual gradient calculations were completely accurate, I implemented a verification script in PyTorch.

I cast the exact same inputs and weights into PyTorch float64 tensors with `requires_grad=True`, ran a forward pass with PyTorch's `F.relu` and `F.cross_entropy`, called `.backward()`, and compared the resulting PyTorch gradients with my custom NumPy gradients:

```
comparisons = [
    ("W1", dL_dw1, grad_w1_torch),
    ("b1", dL_db1, grad_b1_torch),
    ("W2", dL_dw2, grad_w2_torch),
    ("b2", dL_db2, grad_b2_torch),
]
```
You can check the results in the colab notebook. All errors were around $10^{-17}$ to $10^{-18}$, which is within floating point precision limits.
