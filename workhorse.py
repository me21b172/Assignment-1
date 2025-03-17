import numpy as np
import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Layers:
    def __init__(self,input_dim,output_dim,initialisation = "random",activation="sigmoid"): 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        if initialisation == "random":
            self.W = np.random.randn(output_dim,input_dim) 
        if initialisation == "Xavier":
            self.W = np.random.randn(output_dim,input_dim) * np.sqrt(1/input_dim)
        self.b = np.zeros((output_dim,1))
        self.db = np.zeros((output_dim,1))
        self.H = None
        self.A = None
        self.dW = np.zeros((output_dim,input_dim))
        self.dH = None
    def activated_values(self,X):
        if self.activation == "sigmoid":
            A = np.clip(X, -700, 700)
            sig = 1 / (1 + np.exp(-A))
            return sig 
        if self.activation == "ReLU":
            relu = lambda x: x if x > 0 else 0
            return np.vectorize(relu)(X)
        if self.activation == "identity":
            return X 
        if self.activation == "Softmax":
            A = np.clip(X, -700, 700)  # Clipping to avoid overflow
            exps = np.exp(A - np.max(A, axis=0, keepdims=True))
            return exps / np.sum(exps, axis=0, keepdims=True)
        if self.activation == "tanh":
            return np.tanh(X)
        
    def dactivation_da(self,X):
        if self.activation == "sigmoid":
            A = np.clip(X, -500, 500)
            return 1/(1+np.exp(-A)) * (1 - 1/(1+np.exp(-A)))
        if self.activation == "ReLU":
            return np.where(X > 0, 1, 0)
        if self.activation == "identity":
            return np.ones_like(X)
        if self.activation == "tanh":
            return 1 - np.tanh(X)**2

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.uW = []
        self.ub = []
        self.mW = []
        self.mb = []
        self.epoch = 1
        for layer in layers:
            self.uW.append(np.zeros((layer.dW.shape[0],layer.dW.shape[1])))
            self.ub.append(np.zeros((layer.db.shape[0],layer.db.shape[1])))
            self.mW.append(np.zeros((layer.dW.shape[0],layer.dW.shape[1])))
            self.mb.append(np.zeros((layer.db.shape[0],layer.db.shape[1])))
    
    def forward(self,X):
        # X = X.reshape(-1,1)
        self.layers[0].A = self.layers[0].W @ X + self.layers[0].b 
        self.layers[0].H = self.layers[0].activated_values(self.layers[0].A)
        for i in range(1,len(self.layers)):
            self.layers[i].A = self.layers[i].W @ self.layers[i-1].H + self.layers[i].b
            self.layers[i].H = self.layers[i].activated_values(self.layers[i].A)
        return self.layers[i].H
    
    def backPropagation(self,X,Y):
        m = X.shape[0]
        encoded_Y = np.zeros((Y.shape[0],10))
        encoded_Y[np.arange(Y.shape[0]),Y] = 1
        n = len(self.layers)
        for i in range(n):
            self.layers[i].dW = np.zeros_like(self.layers[i].dW) 
            self.layers[i].db = np.zeros_like(self.layers[i].db)
        
        y_pred = self.forward(X.T)
        dL_da = - (encoded_Y.T - y_pred)
        for i in range(n-1,0,-1):
            self.layers[i].dW =  dL_da @(self.layers[i-1].H).T / m
            self.layers[i].db = np.sum(dL_da, axis=1, keepdims=True) / m
            dL_dh = (self.layers[i].W).T @ dL_da
            dL_da = dL_dh * self.layers[i-1].dactivation_da(self.layers[i-1].A)
        self.layers[0].dW = dL_da @ (X) / m
        self.layers[0].db = np.sum(dL_da, axis=1, keepdims=True) / m

    def update_step(self,X,Y,beta ,beta1, beta2 ,epsilon = 1e-6,optimiser = "momentum"):
        if optimiser == "momentum" or optimiser == "sgd":
            self.backPropagation(X,Y)
            for i in range(len(self.layers)):
                self.uW[i] = beta*self.uW[i] + self.layers[i].dW
                self.ub[i] = beta*self.ub[i] + self.layers[i].db
            return self.uW,self.ub
            
        if optimiser == "nag":
            weights,bias = [],[]
            for i in range(len(self.layers)):
                weights.append(self.layers[i].W.copy())
                bias.append(self.layers[i].b.copy())
                self.layers[i].W = self.layers[i].W - beta*self.uW[i]
                self.layers[i].b = self.layers[i].b - beta*self.ub[i]
            self.backPropagation(X,Y)
            for i in range(len(self.layers)):
                self.uW[i] = beta*self.uW[i] + self.layers[i].dW
                self.ub[i] = beta*self.ub[i] + self.layers[i].db
                self.layers[i].W = weights[i]
                self.layers[i].b = bias[i]
            return self.uW,self.ub

        if optimiser == "RMSprop":  
            self.backPropagation(X,Y)
            grad_w,grad_b = [],[]
            for i in range(len(self.layers)):
                self.uW[i] = beta*self.uW[i] + (1-beta)*self.layers[i].dW**2
                self.ub[i] = beta*self.ub[i] + (1-beta)*self.layers[i].db**2
                grad_w.append(self.layers[i].dW/(np.sqrt(self.uW[i])+epsilon))
                grad_b.append(self.layers[i].db/(np.sqrt(self.ub[i])+epsilon))
            return grad_w,grad_b
                
        if optimiser == "adam":
            self.backPropagation(X,Y)
            grad_w,grad_b = [],[]
            for i in range(len(self.layers)):
                self.uW[i] = beta2*self.uW[i] + (1-beta2)*self.layers[i].dW**2
                self.ub[i] = beta2*self.ub[i] + (1-beta2)*self.layers[i].db**2
                self.mW[i] = beta1*self.mW[i] + (1-beta1)*self.layers[i].dW
                self.mb[i] = beta1*self.mb[i] + (1-beta1)*self.layers[i].db
                u_hat_w = self.uW[i]/(1-beta2**self.epoch)  
                u_hat_b = self.ub[i]/(1-beta2**self.epoch)
                m_hat_w = self.mW[i]/(1-beta1**self.epoch)
                m_hat_b = self.mb[i]/(1-beta1**self.epoch)
                self.epoch += 1
                grad_w.append(m_hat_w/(np.sqrt(u_hat_w)+epsilon))
                grad_b.append(m_hat_b/(np.sqrt(u_hat_b)+epsilon))
            return grad_w,grad_b
        
        if optimiser == "nadam":
            self.backPropagation(X,Y)
            grad_w,grad_b = [],[]
            for i in range(len(self.layers)):
                self.uW[i] = beta2*self.uW[i] + (1-beta2)*self.layers[i].dW**2
                self.ub[i] = beta2*self.ub[i] + (1-beta2)*self.layers[i].db**2
                self.mW[i] = beta1*self.mW[i] + (1-beta1)*self.layers[i].dW
                self.mb[i] = beta1*self.mb[i] + (1-beta1)*self.layers[i].db
                u_hat_w = self.uW[i]/(1-beta2**self.epoch)  
                u_hat_b = self.ub[i]/(1-beta2**self.epoch)
                m_hat_w = self.mW[i]/(1-beta1**self.epoch)
                m_hat_b = self.mb[i]/(1-beta1**self.epoch)
                grad_w.append((beta1*m_hat_w + ((1-beta1)/(1-beta1**self.epoch))*(self.layers[i].dW))/(np.sqrt(u_hat_w)+epsilon))
                grad_b.append((beta1*m_hat_b + ((1-beta1)/(1-beta1**self.epoch))*(self.layers[i].db))/(np.sqrt(u_hat_b)+epsilon))
                self.epoch += 1
            return grad_w,grad_b    

    def gradient_descent(self, X, Y, beta, eta, optimiser,beta1,beta2,alpha, epsilon = 1e-6,gradientDescent="Minibatch", batch_size=32):
        
        if gradientDescent == "Vanilla":
            step_w, step_b = self.update_step(X,Y,beta=beta,optimiser = optimiser)
            for i in range(len(self.layers)):
                self.layers[i].W = self.layers[i].W - eta*(step_w[i]) - eta*alpha*self.layers[i].W
                self.layers[i].b = self.layers[i].b - eta*(step_b[i]) - eta*alpha*self.layers[i].b
   
        elif gradientDescent == "Minibatch" or optimiser == "sgd":
            m = X.shape[0]
            if optimiser == "sgd":
                batch_size = 1
            for i in range(0, m, batch_size):
                batchX = X[i:i+batch_size]
                batchY = Y[i:i+batch_size]
                step_w, step_b = self.update_step(batchX, batchY, beta=beta,beta1=beta1, beta2=beta2,epsilon=epsilon, optimiser=optimiser)
                for j in range(len(self.layers)):
                    self.layers[j].W -= eta * step_w[j] + eta*alpha*self.layers[j].W
                    self.layers[j].b -= eta * step_b[j] + eta*alpha*self.layers[j].b
    
    def fit(self,X,y,X_validation,y_validation,beta=0.9,beta1=0.99,beta2=0.99,eta=0.0001,epochs=10,optimiser="nadam",epsilon = 1e-6,alpha=0.5,batch_size=64,gradientDescent="Minibatch",verbose=False,heatmap=False):
        print("Initialising Training with the following parameters:")
        print(f"Epochs: {epochs}, Beta: {beta}, Beta1: {beta1}, Eta: {eta}, Optimiser: {optimiser}, Alpha: {alpha}, Batch Size: {batch_size}, Gradient Descent: {gradientDescent}")
        for epoch in range(1,epochs+1):
            self.gradient_descent(X,y,beta=beta,beta1=beta1,beta2=beta2,eta=eta,optimiser = optimiser,alpha=alpha,epsilon=epsilon,gradientDescent = gradientDescent, batch_size=batch_size)
            y_pred = self.forward(X_validation.T)
            val_accuracy = self.accuracy(y_validation, y_pred)
            val_loss = self.loss(X_validation, y_validation, y_pred)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
            if heatmap:
                y_pred = y_pred.argmax(axis=0)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                cm = confusion_matrix(y_validation, y_pred)

                # Create a heatmap with Seaborn
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=class_names, yticklabels=class_names, linewidths=0.5, linecolor='black')

                plt.xlabel("Predicted Labels", fontsize=12, fontweight='bold')
                plt.ylabel("True Labels", fontsize=12, fontweight='bold')
                plt.title(f"Confusion Matrix (Epoch {epoch})", fontsize=14, fontweight='bold')

                # Save the plot as an image and log it in W&B
                plt.savefig("conf_matrix.png")
                plt.close()

                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "conf_matrix": wandb.Image("conf_matrix.png")  # Log as an image instead of default W&B matrix
                }, step=epoch)

            elif verbose:
                y_pred = y_pred.argmax(axis=0)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    }, step=epoch, commit=True)
            
    
    def accuracy(self,y_truth,y_pred):
        y_pred = y_pred.argmax(axis=0)
        accuracy = np.sum(y_pred==y_truth)/len(y_truth)
        return accuracy
    
    def loss(self,X,Y,y_pred):
        m = X.shape[0]
        encoded_Y = np.zeros((Y.shape[0],10))
        encoded_Y[np.arange(Y.shape[0]),Y] = 1
        if self.layers[-1].activation == "Softmax":
            loss = -np.sum(encoded_Y.T * np.log(y_pred+1e-8))/m
        else:
            loss = np.sum((encoded_Y - y_pred.T)**2)/m
        return loss     
