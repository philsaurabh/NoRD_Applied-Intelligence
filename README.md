# NoRD_Applied-Intelligence
Official code for "NoRD: A Framework for Noise-Resilient Self-Distillation through Relative Supervision" Publsihed at Applied Intelligence Journal.

## Communicated at Applied Intelligence Journal
Currently, a very basic working code is given on Cifar100 only. It's ablation study code. So, some of the losses are commented out. Link to the [article](https://link.springer.com/article/10.1007/s10489-025-06355-y) and full code will be made available after acceptance.

## Installation
### Mandatory Dependencies
* Python> 3.7
* PyTorch >= 1.7.0
* pandas
* scipy
* sklearn
* torchvision
* numpy



## Usage


### To run it
```shell script
python teacher_n.py --arch ResNet34    --lr 0.01 --gpu-id 0
python student_n.py --t-path ./experiments/teacher_ResNet34_seed0/   --s-arch ResNet34    --lr 0.05 --gpu-id 0
```
Models can be downloaded from official Pytorch website or from [SSKD Repository](https://github.com/xuguodong03/SSKD).


## Contact

please contact saurabh_2021cs30@iitp.ac.in for any discrepancy.


## Authors:

* Saurabh Sharma
* Shikhar Singh Lodhi
* Vanshika Srivastava
* Joydeep Chandra

## Sources
[SSKD Repository](https://github.com/xuguodong03/SSKD) is used for the basic architecture.
