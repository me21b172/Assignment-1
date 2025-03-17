from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
from wandb_utils import sweep_config,wandbsweep
from workhorse import Layers,NeuralNetwork

def extract_data(dataset):
    if dataset == "fashion_mnist":
        print("Extracting fashion_MNIST data")
        fashion_mnist_data = fashion_mnist.load_data()
    elif dataset == "mnist":
        print("Extracting MNIST data")
        fashion_mnist_data = mnist.load_data()
    train_data_x,train_data_y, test_data_x, test_data_y = fashion_mnist_data[0][0], fashion_mnist_data[0][1], fashion_mnist_data[1][0], fashion_mnist_data[1][1]
    train_data_x = train_data_x.reshape(train_data_x.shape[0],-1)
    test_data_x = test_data_x.reshape(test_data_x.shape[0],-1)
    train_data_x, validation_data_x, train_data_y, validation_data_y = train_test_split(train_data_x, train_data_y, test_size=0.1)
    #Normalising data
    train_data_x = train_data_x / 255.0
    validation_data_x = validation_data_x / 255.0
    test_data_x = test_data_x / 255.0
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return train_data_x, train_data_y, validation_data_x, validation_data_y, test_data_x, test_data_y, class_names

def creating_stratified_data(train_data_x, train_data_y):
    X_stratified_sample, _, y_stratified_sample, _ = train_test_split(
        train_data_x, train_data_y, 
        train_size=0.05, 
        stratify=train_data_y, 
        random_state=42
    )
    return X_stratified_sample, y_stratified_sample

def fit_best_model(X_train,y_train,X_validation,y_validation,sweep_id=2,loss="cross_entropy"):
    layers = []
    input_dim = 784
    for i in range(sweep_config[sweep_id]["parameters"]["hidden_layers"]["values"][0]):
        layers.append(Layers(input_dim=input_dim, 
                            output_dim=sweep_config[sweep_id]["parameters"]["hidden_size"]["values"][0], 
                            initialisation=sweep_config[sweep_id]["parameters"]["initialisation"]["values"][0], 
                            activation=sweep_config[sweep_id]["parameters"]["activation"]["values"][0]))
        input_dim = sweep_config[sweep_id]["parameters"]["hidden_size"]["values"][0]
    if(loss == "cross_entropy"):
        layers.append(Layers(input_dim, 10, activation="Softmax"))
    else:
        layers.append(Layers(input_dim, 10, activation="identity"))

    model = NeuralNetwork(layers)

    model.fit(
        X=X_train,
        y=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        beta=sweep_config[sweep_id]["parameters"]["beta"]["values"][0],
        beta1= sweep_config[sweep_id]["parameters"]["beta1"]["values"][0],
        beta2= sweep_config[sweep_id]["parameters"]["beta2"]["values"][0],
        eta=sweep_config[sweep_id]["parameters"]["eta"]["values"][0],
        epochs=sweep_config[sweep_id]["parameters"]["epochs"]["values"][0],
        optimiser=sweep_config[sweep_id]["parameters"]["optimiser"]["values"][0],
        alpha=sweep_config[sweep_id]["parameters"]["weight_decay"]["values"][0],
        batch_size=sweep_config[sweep_id]["parameters"]["batch_size"]["values"][0],
        gradientDescent="Minibatch",
        verbose=False
    )
    return model

def fit_custom_model(X_train, y_train, X_validation, y_validation,
                          hidden_layers, hidden_size, initialisation, activation, momentum,
                          beta, beta1, beta2, eta, epochs,epsilon, optimiser, weight_decay, batch_size,
                          loss="cross_entropy", gradientDescent="Minibatch", verbose=False):
    if optimiser == "momentum " or optimiser == "nag":
        beta = momentum
    layers = []
    input_dim = 784
    for _ in range(hidden_layers):
        layers.append(Layers(input_dim=input_dim, 
                             output_dim=hidden_size, 
                             initialisation=initialisation, 
                             activation=activation))
        input_dim = hidden_size
    if loss == "cross_entropy":
        layers.append(Layers(input_dim, 10, activation="Softmax"))
    else:
        layers.append(Layers(input_dim, 10, activation="identity"))

    model = NeuralNetwork(layers)

    model.fit(
        X=X_train,
        y=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        beta=beta,
        beta1=beta1,
        beta2=beta2,
        eta=eta,
        epochs=epochs,
        optimiser=optimiser,
        alpha=weight_decay,
        batch_size=batch_size,
        gradientDescent=gradientDescent,
        epsilon = epsilon,
        verbose=verbose
    )
    return model


