import torch
import math
import sys
import numpy as np


# Inverse sigmoid function on torch tensor
def safe_logit(x):
    return torch.log(x / (1 - x + 1e-6) + 1e-6)


# Project x onto the hyper_cube_bound
def hyper_cube_clip(x, hyper_cube_bound):
    x[x > hyper_cube_bound] = hyper_cube_bound
    x[x < -1 * hyper_cube_bound] = -1 * hyper_cube_bound
    return x

def random_normal(shape):
    """ Returns tensor of given shape with unit normal random numbers. """
    mu = torch.zeros(shape)
    std = torch.ones(shape)
    return torch.normal(mu, std)

class GMN_Layer_Vectorized():
    """ Vectorized implementation of GMN_layer """

    def __init__(self, in_features, num_nodes, side_info_size, context_planes, device='cpu'):
        """
        :param in_features: int, number of input features, i.e. nodes from previous layer
        :param num_nodes: int, number of nodes in this layer
        :param side_info_size: int, length of side channel info, i.e. features
        :param context_planes: int, number of context planes to use, number of actual contexts will be 2^{this number}
        """

        self.in_features = in_features + 1 # +1 for bias
        self.context_dim = 2 ** context_planes
        self.context_planes = context_planes
        self.num_nodes = num_nodes
        self.device = device

        # initialize weights
        # w: float tensor of dims [num_nodes, context_planes, in_features]
        self.w = torch.zeros(self.num_nodes, self.context_dim, self.in_features, device=self.device) + (1/self.in_features)

        # initialize contexts
        # context_vectors: float tensor of dims [num_nodes, context_planes, side_info_size]
        # context_vectors: float tensor of dims [num_nodes, context_planes]
        self.context_vectors = random_normal([num_nodes, context_planes, side_info_size]).to(self.device)
        for i in range(num_nodes):
            for j in range(context_planes):
                self.context_vectors[i, j] /= torch.norm(self.context_vectors[i, j], p=2)
        self.context_biases = random_normal([num_nodes, context_planes]).to(self.device)

        self.bias = math.e / (math.e + 1)

    def __call__(self, z, p):
        return self.forward(p, z)

    def get_contexts(self, z):
        """
        Returns the contexts for each neuron in this layer given side channel information z
        :param z: side channel info, float tensor of dims [side_info_size]
        :return: contexts as long tensor of dims [num_numbers]
        """
        # I think this could be faster if we test all 4 planes in one go.
        result = torch.zeros([self.num_nodes], dtype=torch.long, device=self.device)
        for i in range(self.context_planes):

            a = z[None, :]
            b = self.context_vectors[:, i].t()
            c = self.context_biases[:, i]

            # mask will come out as [1,num_neurons] so we flatten it
            bit = 2 ** i
            mask = torch.mm(a, b).view(-1) >= c
            result += mask * bit
        return result

    def forward(self, z, p):
        """
        Forward example through this layer.
        :param z: side channel info, float tensor of dims [side_info_size]
        :param p: predictions from previous layer, float tensor of dims [input_size]
        :return: tuple containing
            activations, float tensor of dims [num_nodes]
            p, original input probabilities, float tensor of dims [input_size]
            contexts, context selected for each node, int tensor of dims [num_nodes]
        """

        # add the bias
        p = torch.cat((torch.as_tensor([self.bias]).to(device=self.device), p))

        # work out context for each node
        contexts = self.get_contexts(z)

        # compute the function
        a = self.w[range(self.num_nodes), contexts] # a is [nodes, probs_from_previous_layer+1]
        b = safe_logit(p)[:, None] # b is [probs_from_previous_layer+1, 1]
        activations = torch.sigmoid(torch.mm(a, b)).view(-1)

        return activations, p, contexts

    def backward(self, forward, target, learning_rate, hyper_cube_bound = 200):
        """
        Perform an update rule for each node in layer
        :param forward: information from forward pass, tuple containing (activations, input probs, contexts)
        :param target: target label, either 0 or 1
        :param learning_rate: float
        :param hyper_cube_bound: bounds for hyper_cube
        :return: none
        """

        activations, p, contexts = forward

        # epsilon = 1e-6
        # bound = torch.as_tensor(1 - epsilon).to(device=self.device)
        #
        # if target == 0:
        #     loss = -1 * torch.log(torch.clamp(1 - activations + epsilon, max=bound))
        # else:
        #     loss = -1 * torch.log(torch.clamp(activations + epsilon, max=bound))
        #
        # if torch.any(torch.isnan(loss)):
        #     print(target, p, (p / (1 - p + 1e-6)))
        #     sys.exit()

        # print(self.w[range(self.num_nodes), contexts].shape)    #[nodes, features]
        # print(activations.shape)                                #[nodes]
        # print(p.shape)                                          #[features]

        self.w[range(self.num_nodes), contexts] = torch.clamp(self.w[range(self.num_nodes), contexts] - learning_rate *
            torch.mm((activations - target)[:, None], safe_logit(p)[None, :]),
            min=-hyper_cube_bound, max=hyper_cube_bound)


class GMN_layer():
    
    def __init__(self, in_features, num_nodes, side_info_size, num_contexts, device='cpu'):
        """
        :param in_features: number of input features (nodes in previous layer)
        :param num_nodes: number of nodes in this layer
        :param side_info_size: size of side channel (z)
        :param num_contexts: number of contex planes
        """
        self.device = device
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.nodes = [GM_Node(in_features + 1, side_info_size, num_contexts, device=device) for i in range(num_nodes)]
        self.bias = math.e / (math.e + 1) # anything from (epsilon...1-epsilon)/{0.5} will be fine.
        
    def __call__(self, z, p):
        return self.forward(p, z)

    def forward(self, z, p):
        """
        :param z: side_channel data, float tensor of dims [in_features]
        :param p: input probabilities from previous layer, float tensor of dims [input_size]
        :return: array of tuples for each node each containing
            node prediction (float),
            probabilities (as per input),
            selected context
        """

        # add the bias
        p_hat = torch.cat((torch.as_tensor([self.bias]).to(device=self.device), p))

        return [self.nodes[i].forward(p_hat, z) for i in range(len(self.nodes))]
    
    def backward(self, forward, target, learning_rate, hyper_cube_bound=200):
        #forward is an array with each element being a tuple (output, p_hat, context)
        for i in range(len(self.nodes)):
            self.nodes[i].backward(forward[i][0], target, forward[i][1], forward[i][2], learning_rate, hyper_cube_bound=hyper_cube_bound)


class GM_Node():

    def __init__(self, in_features, input_size, num_contexts, init_weights=None, device='cpu'):
        """
        :param in_features: int, size of p from previous layer
        :param input_size: int, size of side_channel information
        :param num_contexts: int, number of context planes
        :param init_weights: initial weights, will be auto-generated if not given
        :param device: device, "cpu"|"cuda"
        """

        self.device = device
        self.in_features = in_features
        self.num_contexts = num_contexts
        self.context_dim = 2 ** num_contexts
        if init_weights:
            if not in_features == len(init_weights):
                raise Exception
            else:
                self.w = init_weights.to(device=self.device)
        else:
            # weights can be initialized to anything, but empirically 1/(neurons in previous layer) works well.
            self.w = torch.zeros(self.context_dim, in_features, device=self.device) + (1/in_features)

        # find a random direction then normalize
        self.context_vectors = [random_normal([input_size]) for i in range(num_contexts)]
        for i in range(len(self.context_vectors)):
            self.context_vectors[i] /= torch.norm(self.context_vectors[i], p=2)

        self.context_biases = random_normal([num_contexts])

        self.context_vectors = torch.stack(self.context_vectors).to(dtype=torch.float, device=self.device)
        self.context_biases = self.context_biases.to(dtype=torch.float, device=self.device)

    def get_context(self, x):
        ret = 0
        for i in range(self.num_contexts):
            if torch.dot(x, self.context_vectors[i]) >= self.context_biases[i]:
                ret = ret + 2 ** i
        return ret

    # Geo_wc(z)(x_t = 1; p_t)
    def forward(self, p, z):
        """
        :param p: input probabilities from previous layer, float tensor of dims [input_size]
        :param z: side_channel data, float tensor of dims [in_features]
        :return: tuple of
            node prediction (float),
            probabilities (as per input)
            selected context
        """
        context = self.get_context(z)
        return torch.sigmoid(torch.dot(self.w[context], safe_logit(p))), p, context

    def backward(self, forward, target, p, context, learning_rate, hyper_cube_bound=200):
        # epsilon = 1e-6
        # if target == 0:
        #     loss = -1 * torch.log(min(1 - forward + epsilon, torch.as_tensor(1 - epsilon).to(device=self.device)))
        # else:
        #     loss = -1 * torch.log(min(forward + epsilon, torch.as_tensor(1 - epsilon).to(device=self.device)))
        #
        # if torch.isnan(loss):
        #     print(target, p, (p / (1 - p + 1e-6)))
        #     sys.exit()

        self.w[context] = hyper_cube_clip(self.w[context] - learning_rate * (forward - target) * safe_logit(p),
                                       hyper_cube_bound)