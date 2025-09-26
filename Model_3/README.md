# Model 3 - Baseline CNN for MNIST

## Overview
This is the baseline model for MNIST digit classification, using a simple CNN architecture with SGD optimizer and no learning rate scheduling. This model serves as the foundation for understanding the impact of various optimizations.

## Model Specifications

### Architecture
- **Model Type**: Convolutional Neural Network
- **Total Parameters**: 7,784
- **Parameter Constraint**: ✓ Under 8,000 parameters

### Network Structure
```
Input: (1, 28, 28)
├── Conv2d(1→7, kernel=3) + BatchNorm2d(7) + ReLU
├── Conv2d(7→7, kernel=3) + BatchNorm2d(7) + ReLU  
├── Conv2d(7→10, kernel=3) + BatchNorm2d(10) + ReLU
├── MaxPool2d(2, 2)
├── Conv2d(10→10, kernel=3) + BatchNorm2d(10) + ReLU
├── Conv2d(10→12, kernel=3) + BatchNorm2d(12) + ReLU
├── MaxPool2d(2, 2)
├── Conv2d(12→16, kernel=3, padding=1) + BatchNorm2d(16) + ReLU
├── Conv2d(16→18, kernel=3, padding=1) + BatchNorm2d(18) + ReLU
├── AdaptiveAvgPool2d(1)
└── Linear(18→10)
```

## Training Configuration

### Hyperparameters
```python
# Optimizer
optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0)

# Loss Function
criterion = CrossEntropyLoss(label_smoothing=0.02)

# Learning Rate Schedule
scheduler = None  # No scheduler used

# Regularization
dropout = None  # No dropout layers

# Training
batch_size = 128
num_epochs = 20
```

## Training Results

### Epoch-wise Performance

| Epoch | Train Acc (%) | Test Acc (%) | Train Loss | Test Loss | Time/Epoch |
|-------|--------------|-------------|------------|-----------|------------|
| 1     | 10.49        | 16.72       | 2.2837     | 2.1721    | ~5s        |
| 2     | 23.67        | 39.07       | 2.1224     | 2.0092    | ~5s        |
| 3     | 39.67        | 57.64       | 2.0002     | 1.8864    | ~5s        |
| 4     | 52.58        | 64.93       | 1.8973     | 1.7728    | ~5s        |
| 5     | 58.20        | 66.19       | 1.7971     | 1.6493    | ~5s        |
| 6     | 60.83        | 67.83       | 1.6975     | 1.5395    | ~5s        |
| 7     | 63.31        | 70.24       | 1.5981     | 1.4271    | ~5s        |
| 8     | 65.90        | 72.33       | 1.5010     | 1.3301    | ~5s        |
| 9     | 68.73        | 74.75       | 1.4079     | 1.2339    | ~5s        |
| 10    | 71.07        | 77.28       | 1.3208     | 1.1494    | ~5s        |
| 11    | 73.42        | 79.64       | 1.2349     | 1.0564    | ~5s        |
| 12    | 75.60        | 82.04       | 1.1517     | 0.9710    | ~5s        |
| 13    | 77.61        | 83.94       | 1.0757     | 0.8937    | ~5s        |
| 14    | 79.67        | 85.61       | 1.0071     | 0.8320    | ~5s        |
| 15    | 81.38        | 87.59       | 0.9428     | 0.7729    | ~5s        |
| 16    | 83.24        | 89.42       | 0.8840     | 0.7138    | ~5s        |
| 17    | 85.16        | 91.14       | 0.8295     | 0.6571    | ~5s        |
| 18    | 86.72        | 92.93       | 0.7778     | 0.6105    | ~5s        |
| 19    | 88.50        | 93.64       | 0.7280     | 0.5592    | ~5s        |
| **20** | **89.76**   | **94.47**   | **0.6811** | **0.5170** | ~5s        |

### Final Performance
- **Training Accuracy**: 89.76%
- **Test Accuracy**: 94.47%
- **Training Loss**: 0.6811
- **Test Loss**: 0.5170
- **Total Training Time**: ~100 seconds

## Performance Analysis

### Learning Characteristics
- **Convergence Rate**: Slow and steady
- **Learning Pattern**: Linear improvement with diminishing returns
- **Plateau Behavior**: Begins to plateau around epoch 15-16

### Milestone Achievements
- **50% Test Accuracy**: Epoch 3
- **70% Test Accuracy**: Epoch 7
- **80% Test Accuracy**: Epoch 11
- **90% Test Accuracy**: Epoch 17
- **95% Test Accuracy**: Never achieved
- **99% Test Accuracy**: Never achieved

## Limitations and Issues

### Key Problems Identified
1. **Slow Convergence**: Required 17 epochs to reach 90% accuracy
2. **Limited Peak Performance**: Could not break 95% accuracy barrier
3. **Fixed Learning Rate**: Constant lr=0.001 limited adaptation during training
4. **Suboptimal Training Dynamics**: No learning rate scheduling meant:
   - Too slow learning in early epochs
   - Inability to fine-tune in later epochs

### Areas for Improvement
- Implement learning rate scheduling (e.g., OneCycleLR)
- Consider regularization techniques (dropout)
- Optimize learning rate value
- Extend training epochs if needed

## Code Implementation

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(7)
        self.conv2 = nn.Conv2d(7, 7, kernel_size=3, bias=False)  
        self.bn2 = nn.BatchNorm2d(7)
        self.conv3 = nn.Conv2d(7, 10, kernel_size=3, bias=False)  
        self.bn3 = nn.BatchNorm2d(10)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(10, 10, kernel_size=3, bias=False)  
        self.bn4 = nn.BatchNorm2d(10)
        self.conv5 = nn.Conv2d(10, 12, kernel_size=3, bias=False)  
        self.bn5 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(12, 16, kernel_size=3, padding=1, bias=False)  
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 18, kernel_size=3, padding=1, bias=False)  
        self.bn7 = nn.BatchNorm2d(18)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(18, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

## Conclusion
This baseline model demonstrates that even a simple CNN with basic SGD optimization can achieve reasonable performance (94.47%) on MNIST. However, the slow convergence and inability to reach higher accuracy levels highlight the importance of modern optimization techniques like learning rate scheduling, which are explored in subsequent model iterations.