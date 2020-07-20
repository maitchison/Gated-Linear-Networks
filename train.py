import torch
import numpy as np
import torchvision.datasets as datasets
import model as M
import time
import argparse

# start
# 1.8 updates per second

def center(data):
    """ Center data so that each example has mean 0."""
    for d in data:
        d -= d.mean()

def validate(network, max_samples=None):
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

def train():

    print("Training...")

    num_classes = 10

    network = [M.GMN(10, [128, 128, 1], 28 ** 2, 4) for i in range(num_classes)]

    training_samples = len(mnist_train)

    for i in range(training_samples):
        if (i + 1) % 10 == 0:
            print("Training: {:.2f}%".format(100 * (i + 1) / training_samples))
        for j in range(len(network)):
            network[j].train_on_sample(mnist_train[i].view(-1), 1 if j == mnist_train_labels[i] else 0,
                                       min(100 / (i + 1), 0.01))

        # quick validation check...
        if (i + 1) % 100 == 0:
            validate(network, 25)

    validate(network)


def load_data():

    DATA_PATH = "data"
    DOWNLOAD = False

    global mnist_train
    global mnist_train_labels
    global mnist_val
    global mnist_val_labels

    mnist_train = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).data.float() / 255
    center(mnist_train)
    mnist_train_labels = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).targets
    mnist_val = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).data.float() / 255
    center(mnist_val)
    mnist_val_labels = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).targets

    return mnist_train, mnist_train_labels, mnist_val, mnist_val_labels

def benchmark():
    """ Perform a quick benchmark. """

    UPDATES = 10
    network = [M.GMN(10, [128, 128, 1], 28 ** 2, 4) for i in range(10)]
    print("Performing Benchmark...")
    start_time = time.time()
    for i in range(UPDATES):
        for j in range(len(network)):
            network[j].train_on_sample(mnist_train[i].view(-1), 1 if j == mnist_train_labels[i] else 0,
                                       min(100 / (i + 1), 0.01))

    time_taken = time.time() - start_time
    updates_per_second = UPDATES / time_taken

    print("Performed {:.1f} updates per second.".format(updates_per_second))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default="train")
    args = parser.parse_args()

    if args.mode == 'train':
        load_data()
        train()
    elif args.mode == "benchmark":
        load_data()
        benchmark()
    else:
        raise Exception(f"Invalid mode {args.mode}")

