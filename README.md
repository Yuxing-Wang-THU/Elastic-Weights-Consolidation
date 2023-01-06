# EWC
Pytorch implementations of Elastic Weights Consolidation (EWC) and "Online" EWC.

## Usage (Linux Platform)

### 1. Install Visdom 

```python
pip install visdom
```

### 2. Run Visdom 

```python
Enter 'visdom' in the command line
```

### 3. Run main.py

#### Without EWC

```python
python main.py
```

#### EWC

```python
python main.py --consolidate 
```

#### "Online" EWC

```python
python main.py --consolidate --online     
```

## Results

### Online EWC

Performance

![image](https://github.com/Yuxing-Wang-THU/EWC/blob/main/result/online-ewc.png)

Loss

![image](https://github.com/Yuxing-Wang-THU/EWC/blob/main/result/online-ewc-loss.png)

## Reference

https://github.com/kuc2477/pytorch-ewc

https://github.com/ruinanzhang/Rotated_MNIST_Continual_Learning

DeepMind's paper [Overcoming Catastrophic Forgetting, PNAS 2017](https://arxiv.org/abs/1612.00796).
