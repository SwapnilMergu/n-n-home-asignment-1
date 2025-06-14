# CS5720 - Neural Networks and Deep Learning  
### Home Assignment 1 – Summer 2025  
**Student Name:** Swapnil Mergu
**Student Id:** 700772464
**University of Central Missouri**  
**Course:** CS5720 Neural Networks and Deep Learning  

---

## Assignment Overview

This assignment is divided into three parts:

1. **Tensor Manipulations & Reshaping**
2. **Loss Functions & Hyperparameter Tuning**
3. **Neural Network Training with TensorBoard**

---

## 1: Tensor Manipulations & Reshaping

### Tasks Completed:
- Created a random tensor with shape **(4, 6)** using TensorFlow.
- Found its **rank** and **shape**.
- Reshaped to **(2, 3, 4)** and transposed to **(3, 2, 4)**.
- Broadcasted a tensor of shape **(1, 4)** and added it.
  
### Outputs:
- **Original Shape:** (4, 6)  
- **Rank:** 2  
- **Reshaped Shape:** (2, 3, 4)  
- **Transposed Shape:** (3, 2, 4)

### Broadcasting Explained:
Tensor shapes are compared element-wise, starting from the trailing dimensions. A dimension of size 1 is expanded to match the other tensor's size in that dimension.

---

## 2: Loss Functions & Hyperparameter Tuning

### Tasks Completed:
- Defined sample true labels and model predictions.
- Calculated:
  - **Mean Squared Error (MSE)**
  - **Categorical Cross-Entropy (CCE)**
- Modified predictions and recalculated losses.
- Plotted a bar chart comparing MSE and CCE.

### Results:
| Prediction Version | MSE Loss | CCE Loss |
|--------------------|----------|----------|
| Initial             | 0.0375   | 3.4340   |
| Modified            | 0.2025   | 3.7121   |

### Insight:
- MSE is more sensitive to small numeric differences.
- Cross-Entropy is more appropriate for classification tasks as it penalizes incorrect class probabilities more heavily.

---

## 3: Neural Network Training with TensorBoard

### Tasks Completed:
- Loaded and preprocessed the **MNIST** dataset.
- Built a simple neural network model.
- Trained the model for **5 epochs** with **TensorBoard logging**.
- Analyzed training and validation metrics using TensorBoard.

### Model Details:
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 5

yaml
Copy
Edit

### Observations:
- **Training Accuracy:** Improved to **0.9318**
- **Validation Accuracy:** Improved to **0.9278**
- **Training Loss:** Decreased to **0.2411**
- **Validation Loss:** Decreased to **0.364**
- Weight and bias histograms remained within expected distributions.

---

### TensorBoard Questions Answered:

**1. What patterns do you observe in the accuracy curves?**  
Initial both the curves show the improvement till 4 epochs and after 4th epochs the validation accuracy curve shows level off compare to training accuracy curve, this pattern shows potential overfitting.

**2. How can you detect overfitting using TensorBoard?**  
TensorBoard helps detect overfitting by comparing training vs. validation trends. 
Overfitting signs: 1. Training loss continues to decrease, but Validation loss increases. 2. Training accuracy continues to increases, but Validation accuracy drops.
TensorBoard plot loss and accuracy for both training and validation, which helps to find the widening gap between them.

**3. What happens if the number of epochs increases?**  
When you increase the number of epochs, the model keeps learning from the training data. But in this case the model shows the potential overfitting and by increasing the epochs model memorizes training data instead of generalizing or learn the pattern. TensorBoard helps to choose the correct number of epochs by monitoring the validation loss.

---
# How to Run

```bash
# 1. Clone the Repository
git clone <n-n-home-asignment-1>
open n-n-home-asignment-1/Home_Assignment_1.ipynb

# 2. Run the each Python Scripts cell
