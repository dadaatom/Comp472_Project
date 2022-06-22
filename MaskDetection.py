# PREPROCESSING IMPORTS #

import os
import splitfolders

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision import datasets, transforms


# TRAINING IMPORTS #

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as fun

# EVALUATION IMPORTS #

import matplotlib.pyplot as plt
import itertools

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# ======================= PREDICTIONS ======================= #

imagePath = "Dataset"
outputPath = "train_test_sets"

trainDir = outputPath + "/train"
valDir = outputPath + "/val"
testDir = outputPath + "/test"

classes = ["Cloth", "N95", "None", "Surgical"]  # Folders should be labeled the same as these classes.

splitfolders.ratio(imagePath, output=outputPath, seed=0, ratio=(.8, 0.1, 0.1))

train_batch_size = 35
val_batch_size = 15
test_batch_size = 120

img_height = 140
img_width = 140

transform = transforms.Compose([transforms.Resize((img_width, img_height)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_ds = datasets.ImageFolder(trainDir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

val_ds = datasets.ImageFolder(valDir, transform=transform)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)

test_ds = datasets.ImageFolder(testDir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

# ======================= MODEL ======================= #

num_epochs = 4
learning_rate = 0.0003

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=90, kernel_size=3, padding=1),
            nn.BatchNorm2d(90),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(26010, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

# ======================= TRAINING ======================= #

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def train(fold, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        #data, labels = data.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Fold/Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                fold,epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(fold, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            test_loss += fun.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    precision = round(precision_score(target, pred, average="weighted", zero_division=0), 2)
    recall = round(recall_score(target, pred, average="weighted", zero_division=0), 2)
    f1 = round(f1_score(target, pred, average="weighted", zero_division=0), 2)
    accuracy = round(accuracy_score(target, pred), 2)

    metrics.append([precision, recall, f1, accuracy])

    print('\nTest set for fold {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(fold, test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def preformKFoldTraining(model, optimizer, numFolds = 10):
    kfold = KFold(n_splits=numFolds, shuffle=True)

    for fold,(train_idx,test_idx) in enumerate(kfold.split(train_ds)):

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(train_ds, batch_size=32, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(train_ds, batch_size=32, sampler=test_subsampler)

        model.apply(reset_weights)

        for epoch in range(1, num_epochs + 1):
            train(fold, model, trainloader, optimizer, epoch)

        test(fold,model, testloader)

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

metrics = []

#preformKFoldTraining(model, optimizer)
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
    #for (images, labels) in zip(imagesList, labelsList):
      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss_list.append(loss.item())

      # Backprop and optimisation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Train accuracy
      total = labels.size(0)
      _, predicted = torch.max(outputs.data, 1)
      correct = (predicted == labels).sum().item()
      acc_list.append(correct / total)

torch.save(model.state_dict(), "TrainedModel_V1")

# ======================= PREDICTIONS ======================= #

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for l, p in zip(labels.numpy(), predicted.numpy()):
            y_true.append(classes[l])
            y_pred.append(classes[p])


# ======================= EVALUATION ======================= #

print("\nCross Validation Metrics")
i = 0
for x in metrics:
    print("K-Fold: " + str(i) + " -> [" + str(x[0]) + " " + str(x[1]) + " " + str(x[2]) + " " + str(x[3]) + "]")
    i+=1
print("\n")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

confusion_matrix(y_true=y_true, y_pred=y_pred)
cm_plot_labels = classes
cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=cm_plot_labels)

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix", cmap=plt.cm.Reds)

precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
accuracy = accuracy_score(y_true, y_pred)

print(f"precision: {precision: .2f}")
print(f"recall: {recall: .2f}")
print(f"f1: {f1: .2f}")
print(f"accuracy: {accuracy: .2f}")