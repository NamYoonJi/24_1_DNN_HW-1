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
   - Includes forward pass(from task 1), backpropagation(names as backward), ReLU & Softmax(from task 1), ReLU derivative, weights, and cross entropy loss.
   - ReLU derivative function is for backpropagation using chain rule.
  
   Calculate gradient with numpy
   ```
   model = Neural_Net_np()
   x = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
   output = model.forward(x)
   y = np.array([[0,1], [1,0]])
   loss = model.cross_entropy_loss(output, y)
   grad_w1 = model.backward(x, y)

   print("Gradient of Loss w.r.t w1:", grad_w1)
   ```
   - First, make model using Neural Net np class.
   - Get the output from model.forward()
   - Calculate the loss with y value and the output
   - Backward pass the loss and get the gradient value


## Task 3
1. Output for Pytorch
   ```
   print(w1_torch)
   print(w2_torch)
   tensor([[0.0439, 0.1263, 0.2244, 0.2644],
        [0.3837, 0.4680, 0.5300, 0.6191],
        [0.7234, 0.8097, 0.8356, 0.9737]], requires_grad=True)
   tensor([[0.2463, 0.2537],
        [0.4748, 0.4252],
        [0.6806, 0.6194],
        [0.7864, 0.9136]], requires_grad=True)
   ```
2. Output for Numpy
   ```
   print(model.w1)
   print(model.w2)

   [[0.05097084 0.18139549 0.31182013 0.44224478]
    [0.46426742 0.57000749 0.67574755 0.78148762]
    [0.87756399 0.95861948 1.03967497 1.12073046]]
   [[0.18933791 0.31066209]
    [0.44997597 0.45002403]
    [0.71061402 0.58938598]
    [0.97125207 0.72874793]]
   ```
---
### Implementation explanation
1. Pytorch

   Forward pass with dropoout(0.4)
   ```
    z1_torch = torch.matmul(x_torch, w1_torch)
    a1_torch = F.dropout(F.relu(z1_torch), p=0.4, training=True) #relu in inside the dropout
    z2_torch = torch.matmul(a1_torch, w2_torch)
    output = F.softmax(z2_torch)
   ```
   - forward pass with matrix multiplication
   - After ReLU activation, randomly dropout the result by using F.dropout
   - After dropping out, pass it through weight 2 and softmax

   Calculate the loss & backpropagate
   ```
   loss = cross_entropy_loss(output, y_torch)
   loss.backward()
   ```
   Update the weights using learning rate and gradient of the loss with respect to w1, and w2
   ```
    with torch.no_grad():
        w1_torch -= learning_rate * w1_torch.grad
        w2_torch -= learning_rate * w2_torch.grad

        w1_torch.grad.zero_()
        w2_torch.grad.zero_()
   ```
   - update with learning rate(0.01) * gradient with respect to each weight matrices
   - initialize the gradient to zero for next iteration
   - set torch.no_grad() since we don't train at updating step 
   
   
2. Numpy
   Update Neural Net np Class
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
        # Gradient of the loss with respect to softmax input
        self.dL_dz2 = self.output - y
        self.dL_dw2 = self.a1.T.dot(self.dL_dz2)
        self.dL_da1 = self.dL_dz2.dot(self.w2.T)
        self.dL_dz1 = self.dL_da1 * self.ReLU_derivative(self.z1)
        self.dL_dw1 = x.T.dot(self.dL_dz1)
        return self.dL_dw1, self.dL_dw2


    def dropout(self, a1, rate=0.4):
        # Generate a mask to drop out neurons
        mask = np.random.binomial(1, 1-rate, size = a1.shape)
        return a1 * mask


    def forward_with_dropout(self, x):
        self.z1 = x.dot(self.w1)
        self.a1 = self.ReLU(self.z1)
        self.a1_dropout = self.dropout(self.a1)
        self.z2 = self.a1_dropout.dot(self.w2)
        self.output = self.softmax(self.z2)
        return self.output

    def update_weight(self, grad_w1, grad_w2, lr=0.01):
        self.w1 -= lr * grad_w1
        self.w2 -= lr * grad_w2
   ```
   - Added dropout, forward with dropout, update weight function.
  
     
     - Dropout function
       ```
        def dropout(self, a1, rate=0.4):
        # Generate a mask to drop out neurons
        mask = np.random.binomial(1, 1-rate, size = a1.shape)
        return a1 * mask
       ```
       
   
   
   
   
 
   
   
   
