import torch
import numpy as np
import torchvision.datasets as datasets
import model as M
import time
import argparse
import pickle
from scipy.ndimage import interpolation


# start
# 1.8 updates per second

def moments(image):
    # From https://fsix.github.io/mnist/Deskewing.html
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew_image(image):
    # From https://fsix.github.io/mnist/Deskewing.html
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

def deskew(data):
    """ Applies deskew operation to each example and returns the corrected dataset
        We assume images are 28x28 represented as np arrange of dims [n,28,28] where n is the number of examples.
    """
    result = torch.zeros_like(data)
    for i in range(len(data)):
        result[i] = torch.tensor(deskew_image(data[i].numpy()))
    return result

def center(data):
    """
    Center data so that each example has mean 0.
    :param data: float tensor of dims [n,28,28]
    :return: centered float tensor of dims [n,28,28]
    """
    result = torch.zeros_like(data)
    for i in range(len(data)):
        result[i] = data[i] - data[i].mean()
    return result

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

    return 100 * (correct / total)

def save_db(db):
    with open(db["filename"], 'w') as f:
        pickle.dump(db, f)

def train():

    print("Training...")

    num_classes = 10

    network = [M.GMN(10, [128, 128, 1], 28 ** 2, 4) for i in range(num_classes)]

    training_samples = len(mnist_train)

    db = {}
    db['val_score'] = []
    db['time_stamp'] = []
    db['iteration'] = []

    for i in range(training_samples):
        if (i + 1) % 100 == 0:
            print(" -training: {:.2f}%".format(100 * (i + 1) / training_samples))
        for j in range(len(network)):
            network[j].train_on_sample(mnist_train[i].view(-1), 1 if j == mnist_train_labels[i] else 0,
                                       min(100 / (i + 1), 0.01))

        # quick validation check...
        if i % 1000 == 0:
            score = validate(network, 100)
            db['time_stamp'].append(time.time())
            print(f" -quick check {score:.1f}%")
            db['val_score'].append(score)
            db['iteration'].append(i)

        if (i + 1) % 10000 == 0:
            score = validate(network, 1000)
            print()
            print(f"Performance: {score:.1f}%")
            print()

    score = validate(network)
    print("*"*60)
    print(f"Final score: {score:.2f}%")
    print("*"*60)
    db['final_score'].append(score)

def load_data():

    DATA_PATH = "data"
    DOWNLOAD = False

    global mnist_train
    global mnist_train_labels
    global mnist_val
    global mnist_val_labels

    print("Loading dataset...")

    mnist_train = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).data.float() / 255
    mnist_train = center(deskew(mnist_train))
    mnist_train_labels = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).targets
    mnist_val = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).data.float() / 255
    mnist_val = center(deskew(mnist_val))
    mnist_val_labels = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).targets

    n = len(mnist_train)
    mu = mnist_train.mean()
    sigma = mnist_train.std()

    print(f" - read {n} images with mu:{mu:.1f} sigma:{sigma:.1f}")

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

