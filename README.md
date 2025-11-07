# MAUGIF
This is official Pytorch implementation of "[MAUGIF: Mechanism-Aware Unsupervised General Image Fusion via Dual Cross-Image Autoencoders]()"
 - 
```
@article{
}
```
## Framework
![image]()

## Recommended Environment
 - [ ] torch  1.13.1
 - [ ] cudatoolkit 11.8
 - [ ] torchvision 0.14.0


## Experiments 
### Dataset & Checkpoints & Results
The checkpoints and results can be in [xxx](). 
Download MSRS dataset from [xxx]() 
If you need to evaluate other datasets, please organize them as follows:
```
├── /dataset
    MSRS/
    ├── test
    │   ├── ir
    │   ├── Segmentation_labels
    │   ├── Segmentation_visualize
    │   └── vi
    └── train
        ├── ir
        ├── Segmentation_labels
        └── vi
    MFD/
    ├── test
    │   ├── ir
    │   ├── Segmentation_labels
    │   ├── Segmentation_visualize
    │   └── vi
    ├── test_day
    │   ├── ir
    │   ├── Segmentation_labels
    │   └── vi
    ├── test_night
    │   ├── ir
    │   ├── Segmentation_labels
    │   └── vi
    └── train
        ├── ir
        ├── Segmentation_labels
        └── vi
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
![image]()
## If this work is helpful to you, please cite it as：
```
@article{
}
```
## Acknowledgements