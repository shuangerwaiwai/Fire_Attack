# Fire_Attack

Ubuntu 10.04

python 3.7.10

pytorch-gpu 1.7.1

pytorchvision 0.8.2

数据集：https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs

预训练模型： resnet18

攻击：FGSM

### 1、模型微调

Training complete in 9m 42s

Best val Acc: 0.973570

### 2、测试集测试

Loss: 0.6564 Acc: 0.8003

在测试集上的准确率是80%

### 3、生成对抗样本

Epsilon: 0      Test Accuracy = 6896 / 8617 = 0.8002785192062203

Epsilon: 0.0002 Test Accuracy = 5096 / 8617 = 0.5913891145410236

Epsilon: 0.0004 Test Accuracy = 3318 / 8617 = 0.3850528025995126

Epsilon: 0.0006 Test Accuracy = 2131 / 8617 = 0.24730184518974122

Epsilon: 0.0008 Test Accuracy = 1399 / 8617 = 0.16235348729256122

Epsilon: 0.001  Test Accuracy = 940 / 8617 = 0.10908668910293606

### 4、进行对抗训练，得到最后的测试集准确率

Loss: 0.3127 Acc: 0.9340

经过对抗训练，准确率提升了13%
