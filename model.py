import torch
import torch.nn as nn
import torch.nn.functional as F
import layer as L
import math

def prod(X):
    y = 1
    for x in X: y*= x
    return y

class CNNExtractor(nn.Module):

    def __init__(self, in_shape, out_size):

        assert in_shape == (28, 28)

        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.projection = nn.Linear(64*3*3, out_size)

    def forward(self, x):
        x = x.view(1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.projection(x.view(-1)))
        return x


class IdentityExtractor(nn.Module):
    def __init__(self, in_shape, out_size):
        assert prod(in_shape) == out_size
        super().__init__()

    def forward(self, x):
        return x.view(-1)

class LinearExtractor(nn.Module):
    def __init__(self, in_shape, out_size):
        super().__init__()
        width, height = in_shape
        self.projection = nn.Linear(width*height, out_size, bias=False)

    def forward(self, x):
        return F.tanh(self.projection(x))

class GMN():
    def __init__(self, num_classes, num_nodes, feature_size, num_contexts, device, feature_mapping='identity',
                 context_smoothing=0, p0='z', context_func="half_space", train_features=False):
        """
        :param num_nodes: array of ints containing number of nodes at each layer, last layer should contain 1 node.
        :param feature_size: int, size of side_channel input.
        :param feature_mapping: string, 'identity', 'linear', 'cnn'
        :param num_contexts: int, number of context planes.
        """
        # note we use the side_channel size as initial layer width so that z can be used as p_0

        layer_func = lambda in_size, out_size : L.GMN_Layer_Vectorized(
            in_size, out_size,
            feature_size, num_contexts, device=device,
            context_smoothing=context_smoothing, context_func=context_func)

        feature_extractor_funcs = {
            'identity': IdentityExtractor,
            'linear': LinearExtractor,
            'cnn': CNNExtractor
        }

        assert feature_mapping in feature_extractor_funcs

        self.feature_extractor = feature_extractor_funcs[feature_mapping](in_shape=(28,28), out_size=feature_size)
        self.feature_mapping = feature_mapping
        self.feature_size = feature_size
        self.p0 = p0
        self.num_classes = num_classes
        self.train_features = train_features

        self.device = device

        # create networks
        self.network = []
        for _ in range(num_classes):
            layers = [layer_func(feature_size, num_nodes[0])] + [
                layer_func(num_nodes[i - 1], num_nodes[i])
                for i in range(1, len(num_nodes))
            ]
            self.network.append(layers)

        if self.train_features:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.80)

        self.feature_extractor.to(device=self.device)

        self.mode='train'
        self.counter = 0

    def predict(self, z, label, lr, apply_update=True):

        if self.mode == "test":
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train()

        z = z.to(device=self.device).detach()
        z = self.feature_extractor(z)
        assert z.shape == (self.feature_size,), \
            f"Invalid feature dims, expecting {(self.feature_size,)} found {z.shape}"

        probs = []

        if self.train_features:
            loss = 0.0
        else:
            loss = None

        for j in range(len(self.network)):
            target = 1 if j == label else 0
            prob = self._train_on_sample(self.network[j], z, target, lr, apply_update=apply_update)
            if loss is not None:
                loss = loss + (prob - target) ** 2
            probs.append(prob.detach())

        probs = torch.stack(probs)

        # train input features, batch size of 1 for the moment...
        if loss is not None:

            if self.counter % 1 == 0:
                self.optimizer.zero_grad()

            loss.backward()

            if (self.counter-1) % 1 == 0:
                self.optimizer.step()

            self.counter += 1

        if apply_update:
            with torch.no_grad():
                for network in self.network:
                    for layer in network:
                        layer.apply_update()

        return probs


    def _train_on_sample(self, layers, z, target, learning_rate, apply_update=True, return_activations=False):

        activations = []

        for i in range(len(layers)):
            if i == 0:
                # get a starting guess for class
                z = z - z.min()
                z = (z / z.max()) * 0.90 + 0.05 # no strong assertions...
                p = z
            else:
                # just a normal layer
                p = forward[0]

            forward = layers[i].forward(z, p, is_test=self.mode=='test')
            if return_activations:
                activations.append(forward[0])
            if apply_update:
                layers[i].backward(forward, target, learning_rate)

        prediction = forward[0][0]

        if return_activations:
            return activations
        else:
            return prediction