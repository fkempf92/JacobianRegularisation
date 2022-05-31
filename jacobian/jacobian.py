import torch
import torch.nn as nn
from skorch.utils import TeeGenerator
from skorch.dataset import unpack_data
from skorch import NeuralNet


class NeuralNetKK(nn.Module):
    """
    Class differentiates between two types of architectures: constant (True)
    and pyramid (False). In the constant case, the number of nodes do not
    change over all hidden layers (=constant), whereas in the case of
    pyramid, the number of nodes is halved in each new hidden layer.py

    Example 4 hidden layers:
        constant: input - 10 - 10 - 10 - 10 - output
        pyramid:  input - 64 - 32 - 16 -  8 - output

    """

    def __init__(self, input_dim, nodes, hidden_layers, activation, d=0.1,
                 const_arch=True, grad_penalty=False, batchnorm=False,
                 dropout=True):
        """

        :param input_dim: int
            input dimension
        :param nodes: int
            number of nodes in first hl
        :param hidden_layers: int
            number of hl
        :param activation: nn.ReLU, nn.LeakyReLU or nn.Tanh()
            activation function, applies to all nodes and layers
        :param const_arch: bool
            defines architecture type (see above)

        """
        super(NeuralNetKK, self).__init__()
        self.const_arch = const_arch
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.nodes = nodes
        self.grad_penalty = grad_penalty
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.d = d
        self.g = None
        # double check inputs
        if not isinstance(const_arch, bool):
            raise ValueError('const_arch must be boolean')
        if not isinstance(input_dim, (int, float)):
            raise ValueError('input_dim must be an int or float')
        if input_dim < 1:
            raise ValueError('input_dim must be at least 1')
        if not isinstance(nodes, (int, float)):
            raise ValueError('nodes must be an int or float')
        if nodes < 1:
            raise ValueError('nodes must be at least 1')
        if not isinstance(hidden_layers, (int, float)):
            raise ValueError('hidden_layers must be an int or float')
        if hidden_layers < 1:
            raise ValueError('hidden_layers must be at least 1')
        if not const_arch:
            if nodes / (2 ** hidden_layers) < 1:
                raise ValueError('nodes too small for number of hidden layers')
        # Architectures:
        if const_arch:  # same number of nodes in all hidden layers
            self.layers = nn.ModuleList()
            for i in range(hidden_layers):
                if i == 0:  # this is for input layer
                    self.layers.append(nn.Linear(input_dim, nodes))
                    self.layers.append(activation)
                    if self.batchnorm:
                        self.layers.append(nn.BatchNorm1d(nodes))
                    if self.dropout:
                        self.layers.append(nn.Dropout(d))
                else:  # all subsequent layers
                    self.layers.append(nn.Linear(nodes, nodes))
                    self.layers.append(activation)
                    if self.batchnorm:
                        self.layers.append(nn.BatchNorm1d(nodes))
                    if self.dropout:
                        self.layers.append(nn.Dropout(d))
            self.layers.append(nn.Linear(nodes, 1))  # output layer

        if not const_arch:  # cone-like architecture
            nodes_arch = [nodes] + [nodes / 2 ** (i + 1) for i in
                                    range(hidden_layers)]
            nodes_arch = list(map(int, nodes_arch))
            self.layers = nn.ModuleList()
            for i in range(hidden_layers):
                if i == 0:  # this is for input layer
                    self.layers.append(nn.Linear(input_dim, nodes_arch[i]))
                    self.layers.append(activation)
                    if self.batchnorm:
                        self.layers.append(nn.BatchNorm1d(nodes_arch[i]))
                    if self.dropout:
                        self.layers.append(nn.Dropout(d))
                else:  # all subsequent layers
                    self.layers.append(nn.Linear(nodes_arch[i - 1],
                                                 nodes_arch[i]))
                    self.layers.append(activation)
                    if self.batchnorm:
                        self.layers.append(nn.BatchNorm1d(nodes_arch[i]))
                    if self.dropout:
                        self.layers.append(nn.Dropout(d))
            self.layers.append(nn.Linear(nodes_arch[hidden_layers - 1], 1))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        We need to return a tuple because we need the weights for future
        custom loss functions!

        :param x:
        :return:
        """

        for layer in self.layers:
            x = layer(x)
        return x
    
    def _get_derivs(self, X):
        X.requires_grad_()
        m = self.model.eval()(X)
        g = torch.autograd.grad(outputs=m,
                                inputs=X,
                                grad_outputs=torch.ones_like(m),
                                retain_graph=True,
                                only_inputs=True,
                                create_graph=True)[0]
        return g.detach().numpy()


class RegularizedNet(NeuralNet):
    """
    Extends the NeuralNet class from skorch by overriding existing
    methods and functions to enable:

    1) weight penalisation: L1, L2, L1+L2
    2) gradient penalisation: L1, L2, L1+L2
    3) weight+gradient penalisation

    Parameters
    ----------
    weight_reg: bool
        Do you want to include weight regularisation?
    w_alpha: float
        weight penalty (notation from sklearn, think lambda)
    w_l1_ratio: float
        ratio of penalty for weights.
        w_l1_ratio = 1 corresponds to LASSO
        w_l1_ratio = 0 corresponds to Ridge
        0 < w_l1_ratio < 1 corresponds to Elastic Net

    jacob_reg: bool
        Do you want to include Jacobian regularisation?
    jacob_type: "element" or "mean"
        Do you with to perform elemen-wise or column mean Jacobian
        regularisation?
    j_alpha: float
        input gradient penalty (notation from sklearn, think lambda)
    j_l1_ratio: float
        ratio of penalty for gradients.
        w_l1_ratio = 1 corresponds to LASSO
        w_l1_ratio = 0 corresponds to Ridge
        0 < w_l1_ratio < 1 corresponds to Elastic Net
    """
    def __init__(self, *args,
                 weight_reg=True,
                 w_alpha=1,
                 w_l1_ratio=1,
                 jacob_reg=False,
                 jacob_type='element',
                 j_alpha=1,
                 j_l1_ratio=1, **kwargs):

        super().__init__(*args, **kwargs)
        self.weight_reg = weight_reg
        self.w_alpha = w_alpha
        self.w_l1_ratio = w_l1_ratio

        self.jacob_reg = jacob_reg
        self.jacob_type = jacob_type
        self.j_alpha = j_alpha
        self.j_l1_ratio = j_l1_ratio

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # 1) Regular MSE
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        # 2) Add weight regularisation?
        if self.weight_reg:
            weights = [w for name, w in self.module_.named_parameters()
                       if 'weight' in name]  # we exclude biases
            l1 = torch.tensor([0], dtype=torch.float32)
            l2 = torch.tensor([0], dtype=torch.float32)
            for p in weights:
                l1 += torch.norm(p, 1)
                l2 += torch.norm(p, 2)
            loss += self.w_l1_ratio * self.w_alpha * l1[0] + \
                0.5 * (1 - self.w_l1_ratio) * self.w_alpha * l2[0]
        return loss

    def train_step_single(self, batch, **fit_params):
        """Compute y_pred, loss value, and update net's gradients.
        The module is set to be in train mode (e.g. dropout is
        applied).
        Parameters
        ----------
        Xi : input data
          A batch of the input data.
        yi : target data
          A batch of the target data.
        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.
        """
        # self.module_.train()
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, y_true=yi, X=Xi, training=True)
        # 3) Jacobian regularisation
        if self.jacob_reg:
            Xi.requires_grad_()
            m = self.module_.model(Xi)
            j = torch.autograd.grad(outputs=m,
                                    inputs=Xi,
                                    grad_outputs=torch.ones_like(m),
                                    retain_graph=True,
                                    only_inputs=True,
                                    create_graph=True)[0]
            if self.jacob_type == 'mean':
                j1 = j.mean(dim=0).norm(1)
                j2 = j.mean(dim=0).norm(2)
            if self.jacob_type == 'element':
                j1 = j.norm(1)
                j2 = j.norm(2)
            loss += self.j_l1_ratio * self.j_alpha * j1 + \
                0.5 * (1 - self.j_l1_ratio) * self.j_alpha * j2
        loss.backward()

        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.module_.named_parameters()),
            X=Xi,
            y=yi
        )

        return {
            'loss': loss,
            'y_pred': y_pred,
        }
