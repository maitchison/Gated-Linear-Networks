import torch
import layer as L

class GMN():
    def __init__(self, num_nodes, input_size, num_contexts):
        """
        :param num_nodes: array of ints containing number of nodes at each layer, last layer should contain 1 node.
        :param input_size: int, size of side_channel input.
        :param num_contexts: int, number of context planes.
        """
        # note we use the side_channel size as initial layer width so that z can be used as p_0
        self.layers = [L.GMN_layer(input_size, num_nodes[0], input_size, num_contexts)]
        self.layers = self.layers + [L.GMN_layer(num_nodes[i - 1], num_nodes[i], input_size, num_contexts) for i in range(1, len(num_nodes))]
    
    def train_on_sample(self, z, target, learning_rate):
        z = z.view(-1)
        with torch.no_grad():
            for i in range(len(self.layers)):
                if i == 0:
                    assert z.min() >= 0 and z.max() <= 1, "Side channel must be in the range [0..1]"
                    forward = self.layers[i].forward(z, z)
                    self.layers[i].backward(forward, target, learning_rate)
                else:
                    p = torch.cat([forward[i][0].unsqueeze(0) for i in range(self.layers[i].in_features)])
                    forward = self.layers[i].forward(z, p)
                    self.layers[i].backward(forward, target, learning_rate)

        prediction = forward[0][0]

        return prediction

    def infer(self, z):
        with torch.no_grad():
            z = z.view(-1)
            for i in range(len(self.layers)):
                if i == 0:
                    forward = self.layers[i].forward(z, None)
                else:
                    p = torch.cat([forward[i][0].unsqueeze(0) for i in range(self.layers[i].in_features)])
                    forward = self.layers[i].forward(z, p)
            return forward[0][0]

        
        