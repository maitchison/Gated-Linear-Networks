import torch
import numpy as np
import torchvision.datasets as datasets
import model as M
import time
import argparse
import pickle
from scipy.ndimage import interpolation


# start
# 1.8 updates per second (128, 128, 1)
# 1.2 updates per second (128, 128,128, 1)
# no_grad
# 3.5 updates per second (128, 128, 1)

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

def brightness_norm(data):
    """
    Norm each example in data so that min = -1, max = 1.
    :param data: float tensor of dims [n,28,28]
    :return: float tensor of dims [n,28,28]
    """
    result = torch.zeros_like(data)
    for i in range(len(data)):
        result[i] = (data[i] - data[i].min())
        result[i] = (result[i] / result[i].max()) * 2 - 1
    return result

def validate(network, max_samples=None, allow_training=False):

    validation_samples = max_samples or len(mnist_val)
    correct = 0
    total = 0

    indexes = list(range(validation_samples))
    np.random.shuffle(indexes)

    for i in indexes[:validation_samples]:
        probs = []
        for j in range(len(network)):
            probs.append(
                network[j].train_on_sample(mnist_val[i].view(-1), 1 if j == mnist_val_labels[i] else 0, 0.001, apply_update=allow_training)
            )
        probs = torch.stack(probs)
        predicted_class = torch.argmax(probs)

        if predicted_class == mnist_val_labels[i]:
            correct = correct + 1
        total = total + 1

    return 100 * (correct / total)

def save_db(db):
    with open(db["filename"], 'wb') as f:
        pickle.dump(db, f)

def create_model(num_classes):
    return [M.GMN(
        ([args.nodes] * args.layers) + [1],
        args.feature_size,
        args.context_planes,
        args.device,
        feature_mapping=args.feature_mapping,
    ) for i in range(num_classes)]


def train(run_name, layers=2):

    print("Training...")

    network = create_model(10)

    training_samples = len(mnist_train)

    db = {}
    db['run_name'] = run_name
    db['val_score'] = []
    db['time_stamp'] = []
    db['iteration'] = []
    db['lr'] = []
    db['train_score'] = []
    db['filename'] = db['run_name']+'.db'
    db['layers'] = layers

    print(f"Starting run {db['run_name']}")

    training_order = list(range(training_samples))
    np.random.shuffle(training_order)

    results = []

    training_accuracy = 0.0

    for iteration, i in enumerate(training_order):

        lr = min(args.lr_scale / (iteration + 1), args.lr_max)

        # log progress
        if (iteration + 1) % 100 == 0:
            percent_complete = 100 * (iteration + 1) / training_samples

            #val_accuracy = validate(network, 100)
            #db['val_score'].append(val_accuracy)
            #print(f" -train_err: {100-training_accuracy:.1f}%  val_err: {100-val_accuracy:.1f}% ({percent_complete:.2f}%)")

            print(f" -train_err: {100-training_accuracy:.1f}% ({percent_complete:.2f}%)")

            db['time_stamp'].append(time.time())
            db['iteration'].append(iteration)
            db['train_score'].append(training_accuracy)
            db['lr'].append(lr)
            save_db(db)

        probs = []
        for j in range(len(network)):
            probs.append(network[j].train_on_sample(mnist_train[i].view(-1), 1 if j == mnist_train_labels[i] else 0, lr))
        probs = torch.stack(probs)
        predicted_class = torch.argmax(probs)
        true_class = mnist_train_labels[i]
        results.append(int(predicted_class == true_class))
        training_accuracy = np.mean(results[-1000:]) * 100

    score = validate(network)
    print("*"*60)
    print(f"Final score: {score:.2f}%")
    print("*"*60)
    db['final_score'] = score
    save_db(db)

def load_data(limit_samples=None):

    DATA_PATH = "data"
    DOWNLOAD = False

    global mnist_train
    global mnist_train_labels
    global mnist_val
    global mnist_val_labels

    print("Loading dataset...")

    mnist_train = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).data.float() / 255
    if limit_samples:
        mnist_train = mnist_train[:limit_samples]
    mnist_train = center(deskew(mnist_train))
    mnist_train_labels = datasets.MNIST(root=DATA_PATH, train=True, download=DOWNLOAD).targets
    mnist_val = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).data.float() / 255
    if limit_samples:
        mnist_val = mnist_val[:limit_samples]
    mnist_val = center(deskew(mnist_val))
    mnist_val_labels = datasets.MNIST(root=DATA_PATH, train=False, download=DOWNLOAD).targets

    n = len(mnist_train)
    mu = mnist_train.mean()
    sigma = mnist_train.std()

    print(f" - read {n} images with mu:{mu:.1f} sigma:{sigma:.1f} and range {mnist_train.min():.2f} to {mnist_train.max():.2f}")

    return mnist_train, mnist_train_labels, mnist_val, mnist_val_labels

def benchmark():
    """ Perform a quick benchmark. """

    UPDATES = 100

    network = create_model(10)

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
    parser.add_argument("--run", default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context_planes", type=int, default=4)
    parser.add_argument("--nodes", type=int, help="number of nodes per layer", default=128)
    parser.add_argument("--lr_max", type=float, default=0.01)
    parser.add_argument("--lr_scale", type=float, help="lr=min(lr_scale/t, lr_max)", default=100)
    parser.add_argument("--feature_mapping", type=str, help="identity|linear|cnn", default='identity')
    parser.add_argument("--feature_size", type=int, help="number of features (d)", default=28*28)



    args = parser.parse_args()

    if args.mode == 'train':
        load_data()
        train(run_name=args.run, layers=args.layers)
    elif args.mode == "benchmark":
        load_data(1000)
        benchmark()
    else:
        raise Exception(f"Invalid mode {args.mode}")

