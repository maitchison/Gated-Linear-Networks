import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import math
import model as M
    

def normalize(data):
    for d in data:
        d = d - d.mean()
    return data


def validate(max_samples=None):
    validation_samples = max_samples or len(mnist_val)
    correct = 0
    total = 0

    indexes = list(range(validation_samples))
    np.random.shuffle(indexes)

    for i in indexes[:validation_samples]:

        prob = torch.cat([network[j].infer(mnist_val[i]).unsqueeze(0) for j in range(len(network))])

        pred = torch.argmax(prob)
        if pred == mnist_val_labels[i]:
            correct = correct + 1
        total = total + 1

    print("Accuracy: {:.2f}%".format(100 * (correct / total)))


DATA_PATH = "data"
DOWNLOAD = False

mnist_train = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).data.float() / 255
mnist_train = normalize(mnist_train)
mnist_train_labels = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).targets
mnist_val = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).data.float() / 255
mnist_val = normalize(mnist_val)
mnist_val_labels = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).targets

num_classes = 10

network = [M.GMN(10, [128, 128, 1], 28**2, 4) for i in range(num_classes)]

training_samples = len(mnist_train)

for i in range(training_samples):
    if (i + 1) % 10 == 0:
        print("Training: {:.2f}%".format(100 * (i + 1) / training_samples))
    for j in range(len(network)):
        network[j].train_on_sample(mnist_train[i].view(-1), 1 if j == mnist_train_labels[i] else 0, min(100 / (i + 1), 0.01))

    # quick validation check...
    if (i+1) % 100 == 0:
        validate(25)

validate()
