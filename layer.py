import torch
import math
import sys
import numpy as np


# Inverse sigmoid function on torch tensor
def logit(x):
    return torch.log(x / (1 - x))

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

    def __init__(self, in_features, num_nodes, side_info_size, context_planes, device='cpu', context_smoothing=0.0, context_func="half_space"):
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
        self.context_smoothing = context_smoothing
        self.context_func = context_func

        weight_init_size = 1.0/(self.in_features) # start with geometric averaging
        context_plane_bias = 0.0 if context_func == "half_space" else 0.01

        # initialize weights
        # w: float tensor of dims [num_nodes, context_planes, in_features]
        self.w = torch.zeros(self.num_nodes, self.context_dim, self.in_features, device=self.device) + weight_init_size

        # initialize contexts
        # context_vectors: float tensor of dims [num_nodes, context_planes, side_info_size]
        # context_biases: float tensor of dims [num_nodes, context_planes]

        # todo: reverse order, make this context_planes, num_nodes... will make it faster.

        self.context_vectors = random_normal([num_nodes, context_planes, side_info_size]).to(self.device)
        for i in range(num_nodes):
            for j in range(context_planes):
                self.context_vectors[i, j] /= torch.norm(self.context_vectors[i, j], p=2)
        self.context_biases = random_normal([num_nodes, context_planes]).to(self.device) * context_plane_bias
        self.context_freq = torch.rand([num_nodes, context_planes]).to(self.device) * 2 + 0.05

        self.bias = math.e / (math.e + 1)

    def __call__(self, z, p):
        return self.forward(z, p)

    def get_contexts(self, z):
        """
        Returns the contexts for each neuron in this layer given side channel information z
        :param z: side channel info, float tensor of dims [side_info_size]
        :return: contexts as long tensor of dims [num_numbers]
        """
        # I think this could be faster if we test all 4 planes in one go.
        result = torch.zeros([self.num_nodes], dtype=torch.long, device=self.device)
        for i in range(self.context_planes):
            # mask will come out as [1,num_neurons] so we flatten it
            bit = 2 ** i

            if self.context_func == "half_space":
                mask = torch.mm(z[None, :], self.context_vectors[:, i].t()).view(-1) >= self.context_biases[:, i]
            elif self.context_func == "periodic":
                d = torch.mm(z[None, :], self.context_vectors[:, i].t()).view(-1) - self.context_biases[:, i]
                mask = (d % self.context_freq[:, i]) > (self.context_freq[:, i] / 2)
            else:
                raise Exception(f"Invalid context function {self.context_func}")
            result += mask * bit
        return result

    def forward(self, z, p, is_test=False):
        """
        Forward example through this layer.
        :param z: side channel info, float tensor of dims [side_info_size]
        :param p: predictions from previous layer, float tensor of dims [input_size]
        :return: tuple containing
            activations, float tensor of dims [num_nodes]
            p, original input log probabilities, float tensor of dims [input_size]
            contexts, context selected for each node, int tensor of dims [num_nodes]
        """

        epsilon = 0.001

        # add the bias
        p = torch.cat((torch.as_tensor([self.bias]).to(device=self.device), p))

        # work out context for each node
        contexts = self.get_contexts(z)

        if not is_test and self.context_smoothing > 0:
            for i in range(len(contexts)):
                if np.random.rand() < self.context_smoothing:
                    contexts[i] = np.random.randint(self.context_dim)


        # compute the function
        w = self.w[range(self.num_nodes), contexts] # a is [nodes, probs_from_previous_layer+1]
        log_p = logit(p)                            # b is [probs_from_previous_layer+1, 1]
        activations = torch.sigmoid(torch.mm(w, log_p[:, None])).view(-1)
        # make sure we don't go out of bounds
        activations = torch.clamp(activations, epsilon, 1-epsilon)

        # note: the algorithm says to clamp the activations here, but I do a safe log so it should be fine.
        return activations, log_p, contexts

    def backward(self, forward, target, learning_rate, hyper_cube_bound = 200):
        """
        Perform an update rule for each node in layer
        :param forward: information from forward pass, tuple containing (activations, input probs, contexts)
        :param target: target label, either 0 or 1
        :param learning_rate: float
        :param hyper_cube_bound: bounds for hyper_cube
        :return: none
        """

        activations, log_p, contexts = forward

        delta = - learning_rate * torch.mm((activations - target)[:, None], log_p[None, :])

        # norm = torch.max(torch.abs(delta))
        # if norm > 1:
        #     print("large update!", norm, delta)

        if torch.any(torch.isnan(delta)):
            print("NaNs in update!")
            print("activations:")
            print(activations)
            print("p:")
            print(log_p)
            sys.exit()

        self.w[range(self.num_nodes), contexts] = torch.clamp(
            self.w[range(self.num_nodes), contexts] + delta,
            min=-hyper_cube_bound, max=hyper_cube_bound
        )

def run_tests():

    # test contexts

    nodes = 4
    context_planes = 2

    layer_func = GMN_Layer_Vectorized

    layer = layer_func(10, nodes, 3, context_planes)

    for _ in range(1000):
        # sample a random point
        z = random_normal([3])
        # test point against nodes
        contexts = layer.get_contexts(z)
        # work out contexts by hand
        for i in range(nodes):
            context = 0
            for j in range(context_planes):
                det = z[0] * layer.context_vectors[i,j,0] + z[1] * layer.context_vectors[i,j,1] + z[2] * layer.context_vectors[i,j,2]
                result = det > layer.context_biases[i,j]
                if result:
                    context += (2**j)
            assert context == contexts[i], f"Incorrect context for {layer_func} at {z}, wanted {context} but found {contexts[i]}"


if __name__ == "__main__":
    print("Testing...")
    run_tests()
    print("Done.")