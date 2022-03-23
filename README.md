# JacobianRegularisation

```math
L_1
```
Implements $L_1$ and $L_2$ Jacobian regularisation, both elementwise and column-mean 

Moreover, the NN is directly trained and tuned with Skorch... hence connection to skorch! 


```python
from jacobian import RegularizedNet

from skorch.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV

# setup early stopping
es = EarlyStopping(patience=10, threshold=1e-4)
# setup neural network
net = RegularizedNet(
          module=NeuralNetKK,
          module__input_dim=input_dim,
          module__nodes=64,
          module__const_arch=False,
          module__hidden_layers=1,
          module__activation=nn.LeakyReLU(),
          module__batchnorm=True,
          module__dropout=True,
          criterion=nn.MSELoss,
          w_alpha=None,
          g_l1_ratio=1,
          grads='norm'
          batch_size=1000,
          max_epochs=200,
          callbacks=[es],
          optimizer=torch.optim.Adam,
          optimizer__lr=1e-4,
          optimizer__weight_decay=0,
          iterator_train__drop_last=True,
          verbose=False
)
# get hyperparameters for given input
def get_tune_params(X):
        nf = X.shape[1]
        # hidden layers
        HL = list(map(int, list(
            np.array(range(int(np.log(nf / 2) / np.log(2)))) + 1)))
        # number of nodes
        U = np.random.uniform(low=np.log(0.5 * nf), high=np.log(1.1 * nf),
                              size=15)
        N = sorted(list(map(int, list(np.exp(U)))))
        # learning rate
        LR = list(10 ** np.random.uniform(low=-5, high=-3, size=15))
        LR = sorted(list(map(lambda x: round(x, 7), LR)))
        # dropout probability
        D = list(np.random.uniform(low=0.05, high=0.25, size=15))
        D = sorted(list(map(lambda x: round(x, 3), D)))
        # Penalty
        P = list(10 ** np.random.uniform(low=-8, high=-4, size=15))
        P = sorted(list(map(lambda x: round(x, 10), P)))
        return HL, N, LR, D, P

# hyperparameters for given X
HL, N, LR, D, P = get_tune_params(X=X)

# setup grid
grid_dict = {'module__hidden_layers': HL,
             'module__const_arch': [True, False],
             'module__nodes': N,
             'module__d': D,
             'g_alpha': P,
             'optimizer__lr': LR}
             
# setup grid search – Example
gs = RandomizedSearchCV(net, 
                        grid_dict, 
                        n_jobs=-1,
                        refit=True, 
                        random_state=123,
                        cv=10)
                        
# fit with skorch, where we assume X_train and y_train are torch tensors
gs.fit(X_train, y_train)
# get best model
gs.best_estimator_
# perform predictions
gs.predict(X_test_

```
