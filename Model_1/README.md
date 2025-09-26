Model 1 - Optimized CNN (Best Performance)
Overview
This is the optimized model that achieves the best performance on MNIST digit classification. By combining OneCycleLR scheduling with a carefully balanced architecture (no dropout), this model reaches 99.48% test accuracy while maintaining under 8,000 parameters.
Model Specifications
Architecture

Model Type: Convolutional Neural Network (Optimized)
Total Parameters: 7,784 ✓
Parameter Constraint: Successfully under 8,000 parameters
Key Design Choice: No dropout - allows model to reach full capacity

Network Structure
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
Training Configuration
Hyperparameters
python# Optimizer
optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0)

# Loss Function
criterion = CrossEntropyLoss(label_smoothing=0.02)

# Learning Rate Schedule - CRITICAL COMPONENT
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.05,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,          # 20% warmup
    div_factor=10,          # initial_lr = 0.005
    final_div_factor=100,   # final_lr = 0.00005
    anneal_strategy="cos"
)

# Regularization
# NO DROPOUT - Key difference from Model 2
# Relies on BatchNorm and label smoothing only

# Training
batch_size = 128
num_epochs = 20
Training Results
Epoch-wise Performance
EpochTrain Acc (%)Test Acc (%)Train LossTest LossKey Milestone180.9797.110.75860.2419Instant 97%!296.4896.910.25640.2350Rapid learning397.6198.900.22100.1862Broke 98%497.9799.070.20620.1785Reached 99% ✓598.1798.700.19950.1847Stabilizing698.4199.220.19080.1677Above 99%798.3899.250.18910.1647Consistent898.5699.210.18450.1644High plateau998.6999.130.18130.1647Stable1098.7099.230.18020.1610Mid-point1198.7699.280.17840.1624Improving1298.8099.310.17580.1583Steady gain1398.8699.310.17350.1583Plateau1498.8999.370.17190.1570Small gain1599.0299.400.16910.1570Broke 99.4% ✓1698.9699.360.17000.1551Slight dip1799.0099.440.16790.1540Recovery1899.0599.460.16590.1540Near peak1999.0899.460.16590.1541Sustained2099.1599.480.16390.1536BEST ✓
Final Performance

Training Accuracy: 99.15%
Test Accuracy: 99.48% ✓
Training Loss: 0.1639
Test Loss: 0.1536
Parameters Used: 7,784 / 8,000 (97.3% of limit)

Performance Comparison
ModelTest AccuracyEpochs to 99%Final LossKey FeatureModel 3 (Baseline)94.47%Never0.5170No schedulerModel 2 (w/ Dropout)~98.5-99%Never/Close~0.20OneCycleLR + DropoutModel 1 (This)99.48%40.1536OneCycleLR, No Dropout
Key Success Factors
1. OneCycleLR Scheduler (Critical)

10x faster convergence: 97% in epoch 1 vs epoch 17+ for baseline
Superior peak performance: Enabled reaching 99.48%
Optimal learning dynamics: Warmup → Peak → Cosine decay

2. No Dropout (Counterintuitive but Effective)

Small model insight: With only 7,784 parameters, dropout was over-regularizing
Sufficient regularization: BatchNorm + label smoothing was enough
Full capacity utilization: Allowed model to use all parameters effectively

3. Label Smoothing (0.02)

Gentle regularization: Prevented overfitting without limiting capacity
Improved generalization: Better test performance without dropout

Code Implementation
pythonclass Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers - no dropout defined
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
        
        # NO DROPOUT LAYERS - KEY DIFFERENCE

    def forward(self, x):
        # Clean forward pass without dropout
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
Achievements
Performance Milestones

✅ Under 8,000 parameters: 7,784
✅ 99%+ accuracy: Achieved by epoch 4
✅ 99.4%+ accuracy: Achieved by epoch 15
✅ Target accuracy: 99.48% final

Training Efficiency

97% accuracy in 1 epoch (vs 17+ for baseline)
99% accuracy in 4 epochs (baseline never reached)
Consistent >99% from epoch 6 onwards

Insights and Lessons Learned

OneCycleLR is transformative: Single most important optimization
Less regularization can be more: Dropout hurt performance on small models
BatchNorm is sufficient: Combined with label smoothing, no need for dropout
Fast convergence is achievable: Right optimizer settings matter more than complex architectures
Parameter efficiency: Strategic architecture design can achieve SOTA with minimal parameters

Reproduction Instructions

Model Setup: Use architecture without dropout layers
Optimizer: SGD with lr=0.001, no weight decay
Scheduler: OneCycleLR with max_lr=0.05, 20% warmup
Loss: CrossEntropyLoss with label_smoothing=0.02
Training: 20 epochs, batch_size=128
Expected Result: 99.4%+ test accuracy

Conclusion
Model 1 represents the optimal configuration for this MNIST classification task. By identifying that dropout was over-regularizing the small model and that OneCycleLR was essential for proper convergence, we achieved:

99.48% test accuracy - exceeding the 99.4% target
7,784 parameters - well under the 8,000 limit
Fast convergence - 99% accuracy in just 4 epochs

This demonstrates that for lightweight models, careful optimization of training dynamics (learning rate scheduling) is more important than aggressive regularization techniques.