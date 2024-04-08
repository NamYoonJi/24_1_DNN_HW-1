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
  
## Task 2
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



   
   
   
   
 
   
   
   
