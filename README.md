# Vision Transformer (ViT) with LoRA-based Adaptation

## Overview
This project fine-tunes a Vision Transformer (ViT-B/16) model with **Low-Rank Adaptation (LoRA)** on the **Tiny-ImageNet dataset**. LoRA enables parameter-efficient fine-tuning by introducing trainable low-rank decompositions into pre-trained models while keeping most parameters frozen.

The implementation:
- Uses `torchvision.models.vit_b_16` as the base model
- Replaces `nn.Linear` layers with custom **LoRA-enhanced layers**
- Optimizes only **LoRA parameters and the final classification head**
- Utilizes **multi-GPU support** and **TensorBoard logging**

## Features
- **LoRA-enhanced Linear Layers**: Efficiently fine-tune ViT without modifying the majority of parameters
- **Tiny-ImageNet Dataset**: Uses **80k training**, **10k validation**, and **10k test images**
- **Adaptive Learning Rate Decay**: Linear LR decay across training steps
- **Multi-GPU Support**: Automatically detects and utilizes multiple GPUs
- **TensorBoard Integration**: Tracks **training loss**, **learning rate**, and **validation accuracy**

## Installation
### Prerequisites
Ensure you have Python 3.8+ and install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib tensorboard
```

## Dataset Preparation
Download and extract the **Tiny-ImageNet dataset**, then set the training directory:
```bash
train_dir = "/kaggle/input/tiny-imagenet/tiny-imagenet-200/tiny-imagenet-200/train"
```

## Training Setup
### Hyperparameters
```python
EPOCHS = 5
BATCH_SIZE = 64  # Adjust based on GPU availability
BASE_LR = 1e-3
WEIGHT_DECAY = 0.03
DROPOUT = 0.1
R_LORA = 4  # LoRA rank
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
```

### Model Modifications
- Loads the pre-trained **ViT-B/16** model
- Modifies the classification head for **Tiny-ImageNet (200 classes)**
- Replaces `nn.Linear` layers with **LoRA-enhanced layers**
- Freezes all layers except **LoRA parameters and the classification head**

```python
# Load pre-trained ViT-B/16 model
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# Modify classification head for Tiny-ImageNet (200 classes)
num_features = model.heads.head.in_features
model.heads.head = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(num_features, 200)
)

# Replace Linear layers with LoRA-enhanced versions
replace_linear_with_lora(model)
mark_lora_and_head_as_trainable(model, head_substring="heads.head", bias="none")
```

### Training Loop
- Uses **CrossEntropyLoss** and **AdamW optimizer**
- Implements **linear learning rate decay**
- Runs on multiple GPUs if available
- Logs metrics to **TensorBoard**

```python
# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log metrics
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)
```

## Running the Model
To start training, simply run:
```bash
python train.py
```

## Monitoring with TensorBoard
Launch TensorBoard to monitor training progress:
```bash
tensorboard --logdir=/kaggle/working/tensorboard_logs
```

## Results & Insights
After training, the model's performance is evaluated on the validation set:
```python
print(f"Validation Accuracy: {val_acc:.2f}%")
```

## Future Improvements
- Experiment with **different LoRA ranks** and **dropout rates**
- Implement **LoRA-aware weight merging** for better inference efficiency
- Extend to other datasets like **CIFAR-100, ImageNet-1k**

## References
- [LoRA: Low-Rank Adaptation of Large Models](https://arxiv.org/abs/2106.09685)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)

