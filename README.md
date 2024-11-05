## Enhancing Feature-aided Data Association Tracking in Passive Sonar Arrays: An Advanced Siamese Network Approach
## BiChannel-SiamDinoNet
---

## Introduction
Feature-aided tracking integrates supplementary features into traditional methods and improves the accuracy of data association methods that rely solely on kinematic measurements. Inspired by the successful application of deep learning, we propose BiChannel-SiamDinoNet, an advanced network derived from the Siamese network and integrated into the joint probability data association framework to calculate feature measurement likelihood. Our method forms an embedding space through the feature structure of acoustic targets, bringing similar targets closer together. This makes the system more robust to variations, capable of capturing complex relationships between measurements and targets, and effectively discriminating discrepancies between them. Additionally, we refine the network’s feature extraction module to address underwater acoustic signals' unique line spectrum and implement the knowledge distillation training method to improve the network's ability to assess consistency between features through local representations.

## Acknowledgements

This project is based on the following open-source project:

- **[Siamese-pytorch](https://github.com/bubbliiiing/Siamese-pytorch)** by **bubbliiiing**. We have referenced and modified the convolutional neural network architectures provided in this project to suit our needs.

## Requirements

python==3.9.18
CUDA==11.3
torch==1.10.1+cu113
torchaudio==0.10.1+cu113
torchvision==0.11.2+cu113

## Datasets
Deepship: **(https://github.com/irfankamboh/DeepShip)**
File Structure:
```
	- dataset:
		-feature1
			-train:
				- ship1:
					- 1_0.npy
					- 1_1.npy
					- 1_2.npy
					...
				- ship2
				- ship3
				...
			-valid
		- feature2
		...
```

## Get Start
### a、Settings
```python
	Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = False
    dataset_path    = "your dataset path"
    features        = "your feature name"
    input_shape     = [399,300]
    train_own_data  = True
    pretrained      = True
    model_path      = ""
    Init_Epoch      = 0
    Epoch           = 100
    batch_size      = 64
    Init_lr         = 1e-3
    Min_lr          = Init_lr * 0.01
    optimizer_type  = "sgd"
    momentum        = 0.9
    weight_decay    = 5e-4
    lr_decay_type   = 'cos'
    save_period     = 10
    save_dir        = 'logs'
    num_workers     = 4
    pretrain_flag   = False
```
### b、Training
```bash
python train_dinotfa.py --config data_aug.yml
```
### c、Predict
When predicting, adjust the ```model_path``` and ```input_shape``` parameters in the ```SiameseV2``` class within the Python file according to the model weights and feature dimensions used.
```bash
python siamese_predict.py --feature1 feature1_path --feature2 feature2_path
```

### Reference
https://github.com/bubbliiiing/Siamese-pytorch

https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning
