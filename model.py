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
        #set bias to 0 so we have less effect on the cosign distance of input
        self.projection = nn.Linear(64*3*3, out_size, bias=False)

    def forward(self, x):
        x = x.view(1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.tanh(self.projection(x.view(-1)))
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
        self.projection = nn.Linear(width*height, out_size)

    def forward(self, x):
        return F.tanh(self.projection(x))

class GMN():
    def __init__(self, num_classes, num_nodes, feature_size, num_contexts, device, feature_mapping='identity',
                 context_smoothing=0, p0='z', encoding="one_hot", context_func="half_space"):
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
        assert encoding in ["one_hot", "binary"]

        self.feature_extractor = feature_extractor_funcs[feature_mapping](in_shape=(28,28), out_size=feature_size)
        self.feature_size = feature_size
        self.p0 = p0
        self.num_classes = num_classes

        self.device = device

        self._encoding = encoding

        # create networks
        networks_needed = num_classes if encoding == "one_hot" else math.ceil(math.log2(num_classes))
        self.network = []
        for i in range(networks_needed):
            layers = [layer_func(feature_size, num_nodes[0])] + [
                layer_func(num_nodes[i - 1], num_nodes[i])
                for i in range(1, len(num_nodes))
            ]
            self.network.append(layers)

        self.mode='train'

    def predict(self, z, label, lr, apply_update=True):

        probs = []
        if self._encoding == "one_hot":
            for j in range(len(self.network)):
                target = 1 if j == label else 0
                probs.append(self._train_on_sample(self.network[j], z, target, lr, apply_update=apply_update))
            probs = torch.stack(probs)
        else:
            probs = torch.ones([self.num_classes], dtype=torch.float)
            for j in range(len(self.network)):
                bit = (2 ** j)
                target = 1 if (label & bit) > 0 else 0
                prob = self._train_on_sample(self.network[j], z, target, lr, apply_update=apply_update)
                for i in range(len(probs)):
                    if i & bit > 0:
                        probs[i] *= prob
                    else:
                        probs[i] *= 1 - prob

        return probs


    def _train_on_sample(self, layers, z, target, learning_rate, apply_update=True, p0_override=None, return_activations=False):

        with torch.no_grad():

            z = self.feature_extractor(z)
            z = z.to(device=self.device)
            assert z.shape == (self.feature_size,), f"Invalid feature dims, expecting {(self.feature_size,)} found {z.shape}"

            activations = []

            for i in range(len(layers)):
                if i == 0:
                    # get a starting guess for class
                    p0 = p0_override or self.p0

                    z_to_p = z.detach().clone()
                    z_to_p -= z_to_p.min()
                    z_to_p = (z_to_p / z_to_p.max()) * 0.90 + 0.05 # no strong assertions...

                    if p0 == "z":
                        p = z_to_p
                    elif p0 == "0":
                        p = torch.zeros_like(z) + 0.5
                    elif p0.isdigit():
                        cycle = str(int(p0)-1)
                        if cycle == '0':
                            cycle = 'z' # default to z on first cycle.

                        initial_activations = self._train_on_sample(
                            layers,
                            z, target, learning_rate, apply_update=False, p0_override=cycle,
                            return_activations=True
                        )[-2] # last layer was single neuron, need penultimate layer.
                        p = z_to_p
                        p[:len(initial_activations)] = initial_activations
                    else:
                        raise Exception(f"Invalid p0 initialization {p0}")
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