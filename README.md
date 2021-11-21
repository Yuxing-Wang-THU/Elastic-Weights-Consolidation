# EWC
Pytorch Implementations of EWC and online EWC

DeepMind's paper [Overcoming Catastrophic Forgetting, PNAS 2017](https://arxiv.org/abs/1612.00796).

### 1. Install Visdom 

pip install visdom

### 2. Run Visdom 

Enter 'visdom' in the command line

### 3. Run main.py

python main.py                            # no ewc

python main.py --consolidate              # ewc

python main.py --consolidate --online     # "Online" ewc

## References (Thanks!):

https://github.com/kuc2477/pytorch-ewc

https://github.com/ruinanzhang/Rotated_MNIST_Continual_Learning
