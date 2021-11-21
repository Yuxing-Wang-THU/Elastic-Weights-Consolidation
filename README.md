# EWC
Pytorch Implementations of EWC and "Online" EWC

DeepMind's paper [Overcoming Catastrophic Forgetting, PNAS 2017](https://arxiv.org/abs/1612.00796).

## Usage

### 1. Install Visdom 

pip install visdom

### 2. Run Visdom 

Enter 'visdom' in the command line

### 3. Run main.py

#### Without EWC

python main.py 

#### EWC

python main.py --consolidate 

#### "Online" EWC

python main.py --consolidate --online     


## Result

### 1. Online EWC

./result/online-ewc.png


## References (Thanks!):

https://github.com/kuc2477/pytorch-ewc

https://github.com/ruinanzhang/Rotated_MNIST_Continual_Learning
