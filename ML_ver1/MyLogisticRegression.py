import numpy as np

class MyLogisticRegression():
    def __init__(self,learning_rate,num_of_iterations):
        self.learning_rate=learning_rate
        self.num_of_iterations=num_of_iterations
        self.losses = []


    
    def sigmoid_function(self, z):
        z = np.clip(z, -500, 500) #to avoid large value overflow
        return 1 / (1 + np.exp(-z))

    
    def compute_loss(self, Y, Y_hat):
        epsilon = 1e-9  # "1e-9 = 0.000000001" to avoid log(0)-> infinty
        loss = -(1/self.m) * np.sum( Y * np.log(Y_hat + epsilon) + (1 - Y) * np.log(1 - Y_hat + epsilon) )
        return loss

    
   
    def fit(self,X,Y):

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y size mismatch")

        self.m,self.n=X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y

        #Gradient descent

        for i in range(self.num_of_iterations):

            Y_hat = self.sigmoid_function(self.X.dot(self.w) + self.b)
            
            loss = self.compute_loss(self.Y, Y_hat)
            self.losses.append(loss)
            self.update_weight(Y_hat)

            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {loss}")



    
    def update_weight(self,Y_hat):
        #derivatives

        dw=(1/self.m) * np.dot(self.X.T,(Y_hat-self.Y))
        db=(1/self.m) * np.sum(Y_hat-self.Y)

        #updating gredient descent

        self.w=self.w-self.learning_rate*dw
        self.b=self.b-self.learning_rate*db

    def predict(self,X):
        
        if X.shape[1] != self.w.shape[0]:
            raise ValueError("Feature size mismatch")

        Y_pred=self.sigmoid_function((X.dot(self.w)+self.b))
        Y_pred=np.where(Y_pred>0.5,1,0)
        return Y_pred
