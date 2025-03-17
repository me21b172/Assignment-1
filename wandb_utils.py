import wandb
from workhorse import NeuralNetwork, Layers

sweep_config = [
# search space for loss as cross entropy
{
    "method": "random",  
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [5,10]},
        "hidden_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "eta": {"values": [1e-3, 1e-4]},
        "beta": {"values": [0.9, 0.99]},
        "beta1": {"values": [0.9, 0.99]},
        "beta2": {"values": [0.9, 0.99]},
        "optimiser": {"values": ["momentum", "nag", "RMSprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "initialisation": {"values": ["random", "Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
    }
},
# restricted search space for loss as cross entropy
{
    "method": "bayes",  
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [10]},
        "hidden_layers": {"values": [4, 5]},
        "hidden_size": {"values": [64, 128, 252]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "eta": {"values": [1e-3, 1e-4, 1e-5]},
        "beta": {"values": [0.99]},
        "beta1": {"values": [0.99]},
        "beta2": {"values": [0.99]},
        "optimiser": {"values": ["momentum", "RMSprop","adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "initialisation": {"values": ["Xavier"]},
        "activation": {"values": ["tanh", "ReLU"]},
    }
},
# Optimal parameters for Cross-entropy loss
{
    "method": "bayes",  
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [20]},
        "hidden_layers": {"values": [5]},
        "hidden_size": {"values": [252]},
        "weight_decay": {"values": [0.5]},
        "eta": {"values": [1e-4]},
        "beta": {"values": [0.99]},
        "beta1": {"values": [0.99]},
        "beta2": {"values": [0.99]},
        "optimiser": {"values": ["nadam"]},
        "batch_size": {"values": [64]},
        "initialisation": {"values": ["Xavier"]},
        "activation": {"values": ["tanh"]},
    }   
},
# restricted search space for loss as MSE
{
    "method": "bayes",  
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [10]},
        "hidden_layers": {"values": [3, 4]},
        "hidden_size": {"values": [32, 64, 128, 252]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "eta": {"values": [1e-3, 1e-4, 1e-5]},
        "beta": {"values": [0.9]},
        "beta1": {"values": [0.99]},
        "beta2": {"values": [0.99]},
        "optimiser": {"values": ["RMSprop","adam", "nadam"]},
        "batch_size": {"values": [16, 32]},
        "initialisation": {"values": ["Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
    }
},
#BEst model for MSE
{
    "method": "bayes",  
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [20]},
        "hidden_layers": {"values": [3]},
        "hidden_size": {"values": [252]},
        "weight_decay": {"values": [0]},
        "eta": {"values": [1e-3]},
        "beta": {"values": [0.9]},
        "beta1": {"values": [0.99]},
        "beta2": {"values": [0.99]},
        "optimiser": {"values": ["nadam"]},
        "batch_size": {"values": [16]},
        "initialisation": {"values": ["Xavier"]},
        "activation": {"values": ["tanh"]},
    }
},
#search space for MNIST dataset
{
    "method": "bayes",  
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [7]},
        "hidden_layers": {"values": [3,4,5]},
        "hidden_size": {"values": [64,128,252]},
        "weight_decay": {"values": [0]},
        "eta": {"values": [1e-3]},
        "beta": {"values": [0.9]},
        "beta1": {"values": [0.99]},
        "beta2": {"values": [0.99]},
        "optimiser": {"values": ["nadam"]},
        "batch_size": {"values": [32]},
        "initialisation": {"values": ["Xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
    }
},
#optimal params for MNIST dataset
{
    "method": "bayes",  
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [10]},
        "hidden_layers": {"values": [4]},
        "hidden_size": {"values": [64]},
        "weight_decay": {"values": [0]},
        "eta": {"values": [1e-3]},
        "beta": {"values": [0.9]},
        "beta1": {"values": [0.99]},
        "beta2": {"values": [0.99]},
        "optimiser": {"values": ["nadam"]},
        "batch_size": {"values": [32]},
        "initialisation": {"values": ["Xavier"]},
        "activation": {"values": ["tanh"]},
    }
},
]

def train(X_train, y_train, X_validation, y_validation,loss = "cross_entropy"):
    wandb.init()
    config = wandb.config
    run_name = (f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}"
                f"_opt_{config.optimiser}")
    
    wandb.run.name = run_name
    wandb.run.save() 
    layers = []
    input_dim = 784

    for i in range(config.hidden_layers):
        layers.append(Layers(input_dim, config.hidden_size, initialisation=config.initialisation, activation=config.activation))
        input_dim = config.hidden_size
    if(loss == "cross_entropy"):
        layers.append(Layers(input_dim, 10, activation="Softmax"))
    else:
        print("Yay idientity is triggered!!!!!!!!!!")
        layers.append(Layers(input_dim, 10, activation="identity"))
    
    model = NeuralNetwork(layers)

    model.fit(
        X=X_train,
        y=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        beta=config.beta,
        beta1= config.beta1,
        beta2= config.beta2,
        eta=config.eta,
        epochs=config.epochs,
        optimiser=config.optimiser,
        alpha=config.weight_decay,
        batch_size=config.batch_size,
        gradientDescent="Minibatch",
        verbose=True
    )

    y_pred = model.forward(X_validation.T)
    accuracy = model.accuracy(y_validation, y_pred)
    loss = model.loss(X_validation, y_validation, y_pred)

    wandb.log({
        "accuracy": accuracy,
    })


    print(f"Accuracy: {accuracy}, Loss: {loss}")

def wandbsweep(X_train, y_train, X_validation, y_validation,loss = "cross_entropy", wandb_project = "NeuralNetwork-Hyperparameter-Tuning",sweep_config_id = 0,counts=10):
    sweep_id = wandb.sweep(sweep_config[sweep_config_id], project=wandb_project)
    wandb.agent(sweep_id, function=lambda: train(X_train, y_train, X_validation, y_validation, loss=loss), count=counts)
    wandb.finish()  



