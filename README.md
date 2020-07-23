# Gausian Gated-Linear-Networks

This project will implement [Gausian-Gated Linear Networks](https://arxiv.org/pdf/2006.05964.pdf) in PyTorch and experiment with their use in [Distributional Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf). Currently it just aims to reproduce some of the results in the GNL paper.

## Usage

The script will train a Gated Linear Network on the MNIST dataset. Each hand-written digit is deskewed, and then
contrast normalized. The number of layers can be specified via --layers. 

### Run Benchmark
```python train.py benchmark```

### Train network
```python train.py train```

## Notes on Performance

Gated Linear networks, due to the conditional execution, are fairly slow. This implementation is current running only on the CPU and is not well optimized. I plan to make some changes in the future that should get the network to be similar in speed to a convolutional network. 

Performance using 10 [128,128,1] networks with 16 contexts (4 hyperplanes)

| Version | Updates per second | Time to train MNIST for 1 epoch |
|---------|--------------------|---------------------------------|
Original implementation by killamocingbird | 1.8 | ~9 hours |  
torch.no_grad() | 3.6 | ~4.5 hours |
vectorized layers  | ? | ? |
GPU implementation  | ? | ? |
 
 ## Notes on Accuracy 
 
The accuracy of this implementation does not yet reproduce the 98% result with a single parse in the paper.

| Version | Validation Score | Notes |
|---------|--------------------|---------------------------------|
Original implementation by killamocingbird | 89.6 |  |  
Added deskew | 94.3 | deskewing makes a big difference.  |
Fixed hyperplanes, and weight initialization | ? | - |
  
