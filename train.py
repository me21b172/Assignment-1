import argparse
import numpy as np
from utilities import extract_data, creating_stratified_data, fit_custom_model
from wandb_utils import wandbsweep

def main(args):
    # Extract dataset and loss from command-line arguments
    dataset = args.dataset
    loss = args.loss

    # Extract and stratify data
    X_train, y_train, X_validation, y_validation, X_test, y_test, class_names = extract_data(dataset=dataset)
    X_stratified, y_stratified = creating_stratified_data(train_data_x=X_train, train_data_y=y_train)
    # Running sweeps with different sweep_config_id values
    # wandbsweep(X_train=X_train,
    #            y_train=y_train,
    #            X_validation=X_validation,
    #            y_validation=y_validation,
    #            loss=loss,
    #            sweep_config_id=6,
    #            wandb_project=args.wandb_project,
    #            counts=1)
    # if loss == "cross_entropy":
    # sweep_config_id = 1 if loss == "cross_entropy" else 3
    # wandbsweep(X_train=X_stratified,
    #         y_train=y_stratified,
    #         X_validation=X_validation,
    #         y_validation=y_validation,
    #         loss=loss,
    #         sweep_config_id = sweep_config_id,
    #         wandb_project=args.wandb_project,
    #         counts=70)
    # sweep_config_id = 2 if loss == "cross_entropy" else 4
    # # print(sweep_config_id)
    # wandbsweep(X_train=X_train,
    #            y_train=y_train,
    #            X_validation=X_validation,
    #            y_validation=y_validation,
    #            loss=loss,
    #            sweep_config_id=sweep_config_id,
    #            wandb_project=args.wandb_project,
    #            counts=1)

    # Fit the model with the provided hyperparameters.
    model = fit_custom_model(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        loss=loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimiser=args.optimizer,
        eta=args.learning_rate,
        momentum=args.momentum,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        initialisation=args.weight_init,
        hidden_layers=args.num_layers,
        hidden_size=args.hidden_size,
        activation=args.activation
    )
    y_pred = model.forward(X_test.T).argmax(axis=0)
    y_true = y_test
    accuracy = np.sum(y_pred==y_true)/len(y_pred)
    print(f"Test accuracy of model {accuracy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a neural network with specified hyperparameters and wandb tracking.")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="NeuralNetwork-Hyperparameter-Tuning",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="me21b172-indian-institute-of-technology-madras",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"],
                        help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=20,
                        help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Batch size used to train neural network")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.99,
                        help="Momentum used by momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.99,
                        help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.99,
                        help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.99,
                        help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001,
                        help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.5,
                        help="Weight decay used by optimizers")
    parser.add_argument("-w_i", "--weight_init", type=str, default="Xavier",
                        choices=["random", "Xavier"],
                        help="Weight initialisation method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=5,
                        help="Number of hidden layers used in feedforward neural network")
    parser.add_argument("-sz", "--hidden_size", type=int, default=252,
                        help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", type=str, default="tanh",
                        choices=["identity", "sigmoid", "tanh", "ReLU"],
                        help="Activation function to use")
    
    args = parser.parse_args()
    main(args)