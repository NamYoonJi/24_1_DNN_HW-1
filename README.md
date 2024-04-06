# 24_1_DNN_HW-1
## Task 1
1. Output for pytorch:
```
  PyTorch Neural Network Output:
  
  output for x1: tensor([0.1324, 0.8676], grad_fn=<SelectBackward0>)
  
  output for x2: tensor([0.0145, 0.9855], grad_fn=<SelectBackward0>)
  ```
2. Output for numpy:
```  
  NumPy Neural Network Output:
  
  output for x1: [0.13238887 0.86761113]
  
  output for x2: [0.01448572 0.98551428]
  ```

---
### Implementation explanation

1. Pytorch

   Given data & weights
   ```
   x_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #concat x1, x2 into one matrix
   w1_torch = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], requires_grad=True)
   w2_torch = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], requires_grad=True) 
   ```
   - set w1_torch, w2_torch to require gradient in order to backpropagate

   Forward pass
   ```
   z1_torch = torch.matmul(x_torch, w1_torch)
   a1_torch = F.relu(z1_torch)
   z2_torch = torch.matmul(a1_torch, w2_torch)
   output_torch = F.softmax(z2_torch, dim=1)
   ```
   - Input x -> hidden layer w1 (with matrix multiplication)
   - Activate through ReLU 
   - After activation, pass into w2  (with matrix multiplication)
   - Activate through softmax
  
2. Numpy

   Given data & weights
   ```
   x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
   w1 = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]])
   w2 = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]])
   ```

   Defining functions required before forward pass
   ```
   def relu(x):
    return np.maximum(0, x)

   def softmax(x):
     exp_x = np.exp(x - np.max(x))
     return exp_x / exp_x.sum(axis=0)
   ```

   Forward pass
   ```
   z1 = x.dot(w1) # Input to hidden
   a1 = relu(z1) # Hidden activation
   z2 = a1.dot(w2) # Hidden to output
   output = softmax(z2.T).T # Output activation
   ```
   - trasnposed z2 to calculate in column
   - transposed after the softmax to show in original form

  
## Task 2
1. Output for pytorch (gradient of w1)
```
tensor([[0.3810, 0.3810, 0.3810, 0.3810],
        [0.4663, 0.4663, 0.4663, 0.4663],
        [0.5516, 0.5516, 0.5516, 0.5516]])
```
2. Output for numpy
```
Gradient of Loss w.r.t w1: [[0.38096682 0.38096682 0.38096682 0.38096682]
 [0.46627936 0.46627936 0.46627936 0.46627936]
 [0.5515919  0.5515919  0.5515919  0.5515919 ]]
```
---
### Implementation explanation

1. Pytorch

   Given data & weight & actual y value
   ```
   x_torch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
   w1_torch = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], equires_grad=True)
   w2_torch = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], requires_grad=True)
   y_torch = torch.tensor([[0,1],[1,0]])
   ```


   Defining Cross Entropy Loss
   ```
   def cross_entropy_loss(x, y):
    delta = 1e-7
    return -torch.sum(y*torch.log(x+delta))
   ```
   - Added some very small delta to prevent 0 going inside the log.
   
   Gradient Calculating
   ```
   for i in range(n_iter):
    Y_pred = forward(x_torch)
    loss = cross_entropy_loss(Y_pred, y_torch)
    loss.backward()
   ```
   - After the forward pass, calculate the loss with cross entropy loss
   - loss.backward for backpropagation (in order to get the gradient)

2. Numpy

   Define a class for convenience
   ```
   class Neural_Net_np:
    def __init__(self):
        self.w1 = np.array([[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8],
               [0.9, 1.0, 1.1, 1.2]])
        self.w2 = np.array([[0.2, 0.3],
               [0.4, 0.5],
               [0.6, 0.7],
               [0.8, 0.9]])

    def ReLU(self, z):
        return np.maximum(0, z)

    def ReLU_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = x.dot(self.w1)
        self.a1 = self.ReLU(self.z1)
        self.z2 = self.a1.dot(self.w2)
        self.a2 = self.softmax(self.z2)
        return self.a2

    def cross_entropy_loss(self, y_pred, y):
        return -1 * np.sum(y * np.log(y_pred))


    def backward(self, x, y):
        # Gradient of the loss
        dL_dz2 = self.a2 - y
        dL_da1 = dL_dz2.dot(self.w2.T)
        dL_dz1 = dL_da1 * self.ReLU_derivative(self.z1)
        dL_dw1 = x.T.dot(dL_dz1)
        return dL_dw1
   ```
   
   
   
   
   
