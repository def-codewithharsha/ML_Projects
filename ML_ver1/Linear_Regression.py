import numpy as np

class LinearRegression():
    def __init__(self,learning_rate,num_of_iterations):
        self.learning_rate=learning_rate
        self.num_of_iterations=num_of_iterations

    def fit(self,X,Y): #formula y=wx+b        
        #matrix (m x n)
        self.m,self.n=X.shape #number of row and columns
        #weight and bais
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y

        #Gradient Descent
        for _ in range(self.num_of_iterations):
            self.update_weights()
        

    def update_weights(self):
        Y_prediction=self.predict(self.X)   #y=wx+b

        #calculate Gradient Descent
        
        #dw=-(2 *(self.X.T).dot(self.Y - Y_prediction))/self.m
        #db= (-2 * np.sum(self.Y-Y_prediction))/self.m    

        dw = (2 * (self.X.T).dot(Y_prediction - self.Y)) / self.m
        db = (2 * np.sum(Y_prediction - self.Y)) / self.m
    
        # updating weights
        self.w=self.w-self.learning_rate*dw
        self.b=self.b-self.learning_rate*db

    def predict(self,X): #y=wx+b
        return X.dot(self.w)+self.b