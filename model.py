import torch
import torch.nn as nn
import torch.nn.functional as F
import layer as L

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
    def __init__(self, num_nodes, feature_size, num_contexts, device, feature_mapping='identity'):
        """
        :param num_nodes: array of ints containing number of nodes at each layer, last layer should contain 1 node.
        :param feature_size: int, size of side_channel input.
        :param feature_mapping: string, 'identity', 'linear', 'cnn'
        :param num_contexts: int, number of context planes.
        """
        # note we use the side_channel size as initial layer width so that z can be used as p_0

        layer_func = L.GMN_Layer_Vectorized

        feature_extractor_funcs = {
            'identity': IdentityExtractor,
            'linear': LinearExtractor,
            'cnn': CNNExtractor
        }

        assert feature_mapping in feature_extractor_funcs

        self.feature_extractor = feature_extractor_funcs[feature_mapping](in_shape=(28,28), out_size=feature_size)
        self.feature_size = feature_size

        self.device = device
        self.layers = [layer_func(feature_size, num_nodes[0], feature_size, num_contexts, device=device)]
        self.layers = self.layers + [
            layer_func(num_nodes[i - 1], num_nodes[i], feature_size, num_contexts, device=device)
            for i in range(1, len(num_nodes))
        ]

    def train_on_sample(self, z, target, learning_rate, apply_update=True):

        with torch.no_grad():

            z = self.feature_extractor(z)
            z = z.to(device=self.device)
            assert z.shape == (self.feature_size,), f"Invalid feature dims, expecting {(self.feature_size,)} found {z.shape}"

            for i in range(len(self.layers)):
                if i == 0:
                    # use normalized z as p_0
                    p_0 = z.detach().clone()
                    p_0 -= p_0.min()
                    p_0 = p_0 / p_0.max() * 0.98 + 0.01

                    forward = self.layers[i].forward(z, p_0)
                    if apply_update:
                        self.layers[i].backward(forward, target, learning_rate)
                else:
                    p = forward[0]
                    forward = self.layers[i].forward(z, p)
                    if apply_update:
                        self.layers[i].backward(forward, target, learning_rate)

        prediction = forward[0][0]

        return prediction

    def infer(self, z):
        """
        Returns probability of example z being in class.
        :param z:
        :return:
        """
        return self.train_on_sample(z, 0, 0, apply_update=False)