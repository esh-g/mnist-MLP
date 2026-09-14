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

### 4. Training on MNIST and Evaluation (Deliverable 1.5)

**Setup**
- Dataset: 10,000 samples from `mnist_test.csv`.
- Split: First 5,000 samples used for training, remaining 5,000 reserved exclusively for testing.
- Hyperparameters: 1,000 steps, Learning rate $\alpha = 0.1$, full-batch gradient descent.

**Training Progress**
The cross-entropy loss decreased steadily from the theoretical random baseline of $\approx 2.35$ down to $0.167$:
- Step 0: Mean Loss = $2.3568$
- Step 50: Mean Loss = $0.7998$
- Step 100: Mean Loss = $0.5351$
- Step 500: Mean Loss = $0.2592$
- Step 990: Mean Loss = $0.1673$

**Final Performance on Unseen Test Data**

- Training Accuracy: $95.56\%$
- Test Accuracy (unseen 5,000 samples): $93.14\%$
- Final Test Loss: $0.3255$

Because the test accuracy is within 2-3% of the training accuracy, the network learned real digit features instead of just memorizing the training images.

### 5. Gradient Mistakes and Debugging Journey (Deliverable 1.6)
During the development of this project, I ran into multiple bugs across the mathematical derivation, NumPy code, and training setup. Here are some of them which were resolved by gemini. (I just asked gemini to create a table of things I asked it help for 😅):

| Area / Mistake | What Went Wrong | How I Caught It / Result | How I Fixed It |
| :--- | :--- | :--- | :--- |
| **Output Error Order** | I subtracted 1 from target class *after* doing `dL_dw2 += a1[i] @ error`. | Target class weights were getting updated using predicted probability $\hat{y}$ instead of error $(\hat{y} - 1)$, pushing weights wrong way. | Moved `error[0, correct_values[i]] -= 1` right after copying `Y_pred[i]` so error vector is isolated first. |
| **Loop Accumulator Leak** | Used `dL_db2` as both the running sum and single sample error vector inside the loop. | Iteration 1 had error from iteration 0 mixed into it, so earlier samples kept contaminating later samples' gradients. | Used a separate isolated `error` array that gets overwritten fresh in each iteration. |
| **Chain Rule & Notation** | In handwritten notes, wrote $\frac{\partial L}{\partial z^{[3]}} \times \frac{\partial L}{\partial W^{[3]}}$ and forgot to sum over the 10 classes. Also had dimension typos like $(10 \times 784)$. | Multiplying two derivatives made no mathematical sense and dimensions did not match hidden layer size (128). | Summed across all 10 output neurons ($\sum \delta_i^{[2]} w_{ji}^{[2]}$) and fixed matrix transpose to $(W^{[2]})^T \in \mathbb{R}^{10 \times 128}$. |
| **Double Normalization** | Divided `mnist` by 255.0 twice in the data loading cell. | Pixel values got scaled down by $255^2 = 65025$, making inputs $\approx 10^{-4}$. Gradients vanished and loss barely moved. | Removed the duplicate division line so pixel values stay in $[0.0, 1.0]$. |
| **Slow Python Loop** | Implemented backpropagation with an explicit `for i in range(number_of_samples)` loop. | Running 5,000 samples for 1,000 steps took way too long (over 15 minutes). | Vectorized all gradient calculations using NumPy matrix multiplications (`calculate_gradients_vectorised`), making each step run in milliseconds. |

### 6. Stretch Goals (Deliverable Stretch Goal)
**1. SGD with Momentum Optimizer**
I implemented SGD with Momentum as a drop-in replacement for standard gradient descent. Instead of updating weights directly with the current gradient, momentum tracks an exponentially decaying velocity vector:

$$v \leftarrow \beta v + \alpha \nabla_\theta L$$$$\theta \leftarrow \theta - v$$

Using $\beta = 0.9$ and $\alpha = 0.1$:

- The loss dropped much faster in the first 50 steps (reaching $0.29$ compared to $0.79$ with standard GD).
- After 1,000 steps, training loss dropped to $0.0051$
- Final training accuracy reached $100\%$, and test accuracy improved to $93.14\%$.

**2. Feature Visualization via Backward Projection**
Out of curiosity, I ran an experiment to see what happens if we pass a one-hot encoded vector representing the digit 8 into the last layer and project it backwards to the first layer:

$$\text{Template} = e_8 \left(W^{[2]}\right)^T \left(W^{[1]}\right)^T$$
While it does not draw a crisp handwritten digit (because the network is discriminative rather than generative), the resulting $28 \times 28$ sensitivity maps reveal:
- Red pixels indicate positive weights where strokes actively boost the class score.
- Blue pixels indicate inhibitory regions where ink decreases the class probability (e.g., the holes inside the loops of an 8).
- Plotting templates for all 10 classes produced recognizable silhouettes of 0s, 1s, 3s, and the distinct two-lobed structure of an 8.
