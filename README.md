# MAUGIF
This is official Pytorch implementation of "[MAUGIF: Mechanism-Aware Unsupervised General Image Fusion via Dual Cross-Image Autoencoders]()"
 - 
```
@article{

}
```
## Framework
![image](./images/ModelArch.png)

## Recommended Environment
 - [ ] torch  1.13.1
 - [ ] cudatoolkit 11.8
 - [ ] torchvision 0.14.0
## The architecture of the project is shown as follows:
```
MAUGIF/
├── HSI-MSI
├── images
│   ├── HMF.png
│   ├── Medical.png
│   ├── MFF.png
│   ├── ModelArch.png
│   └── VIF.png
├── IR-VIS
│   ├── checkpoints
│   ├── model
│   ├── out
│   ├── test_demo.py
│   ├── test.py
│   └── train.py
├── MedicalImage
│   ├── checkpoints
│   ├── model
│   ├── out
│   ├── test_demo.py
│   ├── test.py
│   └── train.py
├── MultiFocus
│   ├── checkpoints
│   ├── model
│   ├── out
│   ├── test_demo.py
│   ├── test.py
│   └── train.py
└── README.md
```


## Experiments 
### Dataset & Checkpoints & Results
The checkpoints and results can be in [xxx](). 
Download MSRS dataset from [xxx]() 
If you need to evaluate other datasets, please organize them as follows:
```
├── /dataset
|
|───VIF/
|   ├── test
|   │   ├── ir
|   │   └── vi
|   └── train
|        ├── ir
|        └── vi
|───MFF/
|   ├── test
|   │   ├── FAR
|   │   └── NEAR
|   └── train
|        ├── FAR
|        └── NEAR
└───MEF/
    ├── CT_MRI
    │   ├── MRI
    │   └── CT
    |── PET_MRI
    |    ├── MRI
    |    └── PET
    └── SPECT_MRI
         ├── MRI
         └── SPECT

    ......
```
### Evaluate model
python
```
python test_model.py
```
### run sample
python
```
python test_demo.py --img="./images/00131D_vi.png" --ir="./images/00131D_ir.png" --checkpoint="xxx.pth"
```
### To Train
Before training MAUGIF, you need to download the MSRS dataset MSRS and putting it in ./datasets.

Then running 
python
```
python train_model.py
```
### Fusion comparison
![image](./images/HMF.png)
![image](./images/VIF.png)
![image](./images/MFF.png)
![image](./images/Medical.png)
## If this work is helpful to you, please cite it as：
```
@article{
}
```
## Acknowledgements