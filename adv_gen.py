from __future__  import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import os
# epsilons = [0, .02, .04, .06, .08, .1, .2, .3]
epsilons = [0, .0002, .0004, .0006, .0008, .001]

pretrained_model = "my_model.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = './FLAME_Data'
save_adv_dir = './FLAME_Data/adv'

input_size = 224

batch_size = 1

model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.to(device)
model_ft.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
model_ft.eval()

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['test']}

criterion = nn.CrossEntropyLoss()

def fgsm_attack(image, eposilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + eposilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def get_adv(model, device, test_loader, epsilon, criterion):
    
    correct = 0,
    correct = int(correct[0])
    adv_examples = []

    adv_image_count = 0,
    adv_image_count = int(adv_image_count[0])

    for data, target in test_loader['test']:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        outputs = model(data)

        _, init_preds = torch.max(outputs, 1)
        if init_preds != target.data:
            continue

        loss = criterion(outputs, target)
        # loss = F.nll_loss(outputs, target)

        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)

        _, final_pred = torch.max(output, 1)

        if final_pred == target.data:
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                data = data.squeeze().detach().cpu().numpy()
                adv_examples.append((data, init_preds, final_pred, adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                data = data.squeeze().detach().cpu().numpy()
                adv_examples.append((data, init_preds, final_pred, adv_ex))

            if epsilon == 0.0002:
                perturbed_data = perturbed_data.squeeze()
                perturbed_image = transforms.ToPILImage()(perturbed_data).convert('RGB')
                if target.item() == 1:
                    perturbed_image.save(os.path.join(save_adv_dir, 'Fire', 'adv' + str(adv_image_count) + '.png'))
                else:
                    perturbed_image.save(os.path.join(save_adv_dir, 'No_Fire', 'adv' + str(adv_image_count) + '.png'))
                adv_image_count += 1

    final_acc = correct/float(len(test_loader['test']))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader['test']), final_acc))

    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = get_adv(model_ft, device, dataloaders_dict, eps, criterion)
    accuracies.append(acc)
    examples.append(ex)

cnt = 0
plt.figure(figsize=(50, 20))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]*2), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        image,orig,adv,ex = examples[i][j]
        plt.title("origin")
        plt.imshow(image.transpose([1, 2, 0]))
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]*2), cnt)
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex.transpose([1, 2, 0]))
plt.tight_layout()
# plt.show()
plt.savefig("someadv.png")



