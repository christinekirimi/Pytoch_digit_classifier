
# PyTorch Beginner's Toolkit – MNIST Image Classification

A **beginner-friendly, AI-assisted** deep learning project using **PyTorch** to classify handwritten digits from the **MNIST dataset**.
This project was built as part of the **Moringa AI Capstone: Beginner’s Toolkit with GenAI** challenge.

---

## 📌 Project Overview

This repository demonstrates:

* Setting up a PyTorch development environment.
* Understanding tensors, autograd, and neural network basics.
* Building and training a fully connected neural network for MNIST digit classification.
* Using AI prompts to accelerate learning and debugging.

**End Goal:** Achieve >90% accuracy on the MNIST test set.

---

## 🛠 Tech Stack

* **Language:** Python 3.8+
* **Library:** PyTorch
* **Dataset:** MNIST (via `torchvision.datasets`)
* **IDE:** VS Code / Jupyter Notebook
* **AI Assistance:** Used AI prompts to guide learning and code optimization.

---

## 📂 Project Structure

```
├── Gen_AI.ipynb                # Main training notebook
├── mnist_model.pth             # Saved trained model
├── training_progress.png       # Training loss & accuracy graph
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/pytorch-beginner-toolkit.git
cd pytorch-beginner-toolkit
```

2. **Create a virtual environment** (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify PyTorch installation**

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## 🚀 Usage

### 1. Run the training notebook

Open `Gen_AI.ipynb` in Jupyter Notebook or VS Code and run all cells.

The notebook will:

* Download and preprocess the MNIST dataset.
* Define and train the neural network.
* Evaluate performance on the test set.
* Save the trained model (`mnist_model.pth`).

### 2. Predict a single digit

At the end of the notebook, you can run the **prediction function** to classify a single test image and display the probability distribution.

---

## 📊 Example Output

```
Using device: cpu
Starting training...
Epoch 1 - Loss: 0.42 - Accuracy: 90.32%
...
Model saved as 'mnist_model.pth'
Predicted digit: 7
```

![Training Progress](training_progress.png)

---

## 🧠 Learning Journey (AI Prompt Journal)

| Day       | Focus     | Prompt Summary                       | Key Takeaway                     |
| --------- | --------- | ------------------------------------ | -------------------------------- |
| Monday    | Setup     | How to install and configure PyTorch | Environment ready quickly        |
| Tuesday   | Basics    | Tensors, autograd, simple NN         | Understood PyTorch core concepts |
| Wednesday | Project   | Full MNIST classifier                | End-to-end working code          |
| Wednesday | Debugging | Improve training speed/accuracy      | Added dropout, Adam optimizer    |

---

## 🛠 Common Issues & Fixes

* **Device mismatch error** – Ensure both model and data are moved to the same device (`cpu` or `cuda`).
* **Loss not decreasing** – Normalize data, lower learning rate, or switch optimizer to Adam.
* **Memory errors** – Reduce batch size.

---

## 📚 References

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [Fast.ai Practical Deep Learning](https://course.fast.ai/)

---

## 📌 Next Steps

* Implement Convolutional Neural Networks (CNNs).
* Apply transfer learning with pre-trained models.
* Train on GPU for faster convergence.
* Deploy the model using TorchScript.

