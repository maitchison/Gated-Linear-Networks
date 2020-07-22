import torch
import math
import torch.distributions as tdist

normal_distribution = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.1]))

class GMN_Layer_Vectorized():
    """ Vectorized implementation of GMN_layer """

    def __init__(self, in_features, num_nodes, side_info_size, num_contexts):
        """
        :param in_features: int, number of input features, i.e. nodes from previous layer
        :param num_nodes: int, number of nodes in this layer
        :param side_info_size: int, length of side channel info, i.e. features
        :param num_contexts: int, number of context planes to use, number of actual contexts will be 2^{this number}
        """

        self.in_features = in_features + 1 # +1 for bias
        self.context_dim = 2 ** num_contexts
        self.num_nodes = num_nodes

        # generate vectorized
        self.w = torch.zeros(self.num_nodes, self.context_dim, self.in_features)
        self.context_vectors = torch.Tensor(
            [[normal_distribution.sample([side_info_size]).view(-1) for i in range(num_contexts)] for _ in range(self.num_nodes)]
        )
        self.context_biases = torch.zeros(self.num_nodes, self.num_contexts)

        self.bias = math.e / (math.e + 1)

    def __call__(self, z, p):
        return self.forward(p, z)

    def get_context(self, x):
        ret = 0
        for i in range(self.num_contexts):
            if torch.dot(x, self.context_vectors[i]) >= self.context_biases[i]:
                ret = ret + 2 ** i
        return ret

    def forward(self, z, p):
        if not p is None:
            p_hat = torch.cat((torch.as_tensor([self.bias]), p))
        else:
            #Forward with random base probabilities
            p_hat = torch.cat((torch.as_tensor([self.bias]), 0.5 * torch.ones(self.in_features)))

        #[self.nodes[i].forward(p_hat, z) for i in range(len(self.nodes))]
        #context = self.get_context(z)
        #return torch.sigmoid(torch.dot(self.w[context], GM_Node.logit(p))), p, context

        pass

    def backward(self, forward, target, learning_rate, hyper_cube_bound = 200):
        #forward is an array with each element being a tuple (output, p_hat, context)
        loss = []

        pass

        #for i in range(len(self.nodes)):
        #    loss.append(self.nodes[i].backward(forward[i][0], target, forward[i][1], forward[i][2], learning_rate, hyper_cube_bound=hyper_cube_bound))

        return loss



class GMN_layer():
    
    def __init__(self, in_features, num_nodes, side_info_size, num_contexts):
        self.in_features = in_features
        self.nodes = [GM_Node(in_features + 1, side_info_size, num_contexts) for i in range(num_nodes)]
        self.bias = math.e / (math.e + 1)
        
    def __call__(self, z, p):
        return self.forward(p, z)

    def forward(self, z, p):
        if not p is None:
            p_hat = torch.cat((torch.as_tensor([self.bias]), p))
        else:
            #Forward with random base probabilities
            p_hat = torch.cat((torch.as_tensor([self.bias]), 0.5 * torch.ones(self.in_features)))

        return [self.nodes[i].forward(p_hat, z) for i in range(len(self.nodes))]
    
    def backward(self, forward, target, learning_rate, hyper_cube_bound = 200):
        #forward is an array with each element being a tuple (output, p_hat, context)
        loss = []
        for i in range(len(self.nodes)):
            loss.append(self.nodes[i].backward(forward[i][0], target, forward[i][1], forward[i][2], learning_rate, hyper_cube_bound=hyper_cube_bound))
        return loss


class GM_Node():

    def __init__(self, in_features, input_size, num_contexts, init_weights=None):
        self.in_features = in_features
        self.num_contexts = num_contexts
        self.context_dim = 2 ** num_contexts
        if init_weights:
            if not in_features == len(init_weights):
                raise Exception
            else:
                self.w = init_weights
        else:
            self.w = torch.zeros(self.context_dim, in_features)

        self.context_vectors = [normal_distribution.sample([input_size]).view(-1) for i in range(num_contexts)]
        self.context_biases = [0.0 for i in range(num_contexts)]

    def get_context(self, x):
        ret = 0
        for i in range(self.num_contexts):
            if torch.dot(x, self.context_vectors[i]) >= self.context_biases[i]:
                ret = ret + 2 ** i
        return ret

    # Geo_wc(z)(x_t = 1; p_t)
    def forward(self, p, z):
        context = self.get_context(z)
        return torch.sigmoid(torch.dot(self.w[context], GM_Node.logit(p))), p, context

    def backward(self, forward, target, p, context, learning_rate, hyper_cube_bound=200):
        epsilon = 1e-6
        if target == 0:
            loss = -1 * torch.log(min(1 - forward + epsilon, torch.as_tensor(1 - epsilon)))
        else:
            loss = -1 * torch.log(min(forward + epsilon, torch.as_tensor(1 - epsilon)))

        if torch.isnan(loss):
            print(target, p, (p / (1 - p + 1e-6)))
            sys.exit()

        self.w[context] = GM_Node.clip(self.w[context] - learning_rate * (forward - target) * GM_Node.logit(p),
                                       hyper_cube_bound)
        return loss

    # Inverse sigmoid function on torch tensor
    def logit(x):
        return torch.log(x / (1 - x + 1e-6) + 1e-6)

    # Project x onto the hyper_cube_bound
    def clip(x, hyper_cube_bound):
        x[x > hyper_cube_bound] = hyper_cube_bound
        x[x < -1 * hyper_cube_bound] = -1 * hyper_cube_bound
        return x


