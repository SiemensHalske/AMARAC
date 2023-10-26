import keras.optimizers as optimizers


class optimizer_adam:
    def __init__(self, learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False, name='Adam'):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad
        self.name = name
        self.optimizer = optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            amsgrad=self.amsgrad,
            name=self.name
        )


class optimizer_sgd:
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = name
        self.optimizer = optimizers.SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            name=self.name
        )

        return self.optimizer


class optimizer_rmsprop:
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop'):
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered
        self.name = name
        self.optimizer = optimizers.RMSprop(
            learning_rate=self.learning_rate,
            rho=self.rho,
            momentum=self.momentum,
            epsilon=self.epsilon,
            centered=self.centered,
            name=self.name
        )

        return self.optimizer
