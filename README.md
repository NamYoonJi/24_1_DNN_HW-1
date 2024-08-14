# 24_1_DNN_HW-1 Pytorch tutorial

This homework assignment aims to implement a neural network using two popular packages:
NumPy(>1.20) and PyTorch(>2.2.1). The implementation should be in Python and the resulting
code should be submitted along with a detailed report that explains the implementation.


## Task 1 

Given a neural network architecture and a pair of inputs with weights, implement neural
networks using NumPy and PyTorch.


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
  
## Task 2

Print out gradients of loss with respect to a specific weight from two neural networks. The neural
networks are the ones you implemented in Task 1. The loss function is cross-entropy loss. We also note that cross-entropies here should be implemented by your hand. 

Print the gradients of the loss function with respect to weight w1 for both neural networks, which are from both NumPy and PyTorch.

1. Output for pytorch (gradient of w1)
```
tensor([[0.3810, 0.3810, 0.3810, 0.3810],
        [0.4663, 0.4663, 0.4663, 0.4663],
        [0.5516, 0.5516, 0.5516, 0.5516]])
```
2. Output for numpy
```
Gradient of Loss with respect to w1:
 [[0.38096682 0.38096682 0.38096682 0.38096682]
 [0.46627936 0.46627936 0.46627936 0.46627936]
 [0.5515919  0.5515919  0.5515919  0.5515919 ]]
```
---

## Task 3

Please repeat the processes from Task 2 and update the w1 and w2 FOR 100 times (i.e., 100
epochs), applying a dropout rate of 0.4. In your report, provide the updated w1 and w2 for both
NumPy and PyTorch


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



   
   
   
   
 
   
   
   
