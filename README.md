# Variational Dropout Sparsifies Deep Neural Networks

This repo contains the code for the ICML17 paper, [Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/abs/1701.05369).

*Extension on ResNet* 

*Fixed the out-dated dependency*

## MNIST Experiments 

The table containes the comparison of different sparsity-inducing techniques (Pruning (Han et al., 2015b;a), DNS (Guo et al., 2016), SWS (Ullrich et al., 2017)) on LeNet architectures.
VD method provides the highest level of sparsity with a similar accuracy

| Network       | Method   | Error | Sparsity per Layer  |  Compression |
| -------------: | -------- | ----- | ------------------- | :--------------: |
|               | Original | 1.64  |                     | 1              |
|               | Pruning  | 1.59  | 92.0 − 91.0 − 74.0  | 12             |
| LeNet-300-100 | DNS      | 1.99  | 98.2 − 98.2 − 94.5  | 56             |
|               | SWS      | 1.94  |                     | 23             |
| (ours)        | SparseVD | 1.92  | 98.9 − 97.2 − 62.0  | **68**         |
||||||
|               | Original | 0.8   |                     | 1              |
|               | Pruning  | 0.77  | 34 − 88 − 92.0 − 81 | 12             |
| LeNet-5       | DNS      | 0.91  | 86 − 97 − 99.3 − 96 | 111            |
|               | SWH      | 0.97  |                     | 200            |
| (ours)        | SparseVD | 0.75  | 67 − 98 − 99.8 − 95 | **280**        |

# Environment setup

```(bash)
sudo apt install virtualenv python-pip python-dev
virtualenv venv --system-site-packages
source venv/bin/activate

pip install numpy tabulate 'ipython[all]' sklearn matplotlib seaborn  
pip install --upgrade https://github.com/Theano/Theano/archive/rel-1.0.0.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

# Launch experiments 

```(bash)
source ~/venv/bin/activate
cd variational-dropout-sparsifies-dnn
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' ipython ./experiments/<experiment>.py
```
Variational Dropout on ResNet
```
./ResNet/run.sh
```
PS: If you have CuDNN problem please look at this [issue](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn/issues/3).

# Citation

If you found this code useful please cite the paper of the original authors 

```
@article{molchanov2017variational,
  title={Variational Dropout Sparsifies Deep Neural Networks},
  author={Molchanov, Dmitry and Ashukha, Arsenii and Vetrov, Dmitry},
  journal={arXiv preprint arXiv:1701.05369},
  year={2017}
}
```
