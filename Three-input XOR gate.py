import numpy as np

x = np.array(([0,0,0],[0,0,1],[0,1,0],\
              [0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]),dtype=float)

y = np.array(([1],[0],[0],[0],\
              [0],[0],[0],[1]),dtype=float)


x_predicted = np.array((1,1,1),dtype=float)
x = x/np.amax(x,axis=0)
x_predicted_2=x_predicted/np.amax(x_predicted,axis=0)

lossfile = open("Projects/Machine Learning/XOR Gate/SumSquaredLossList.csv1","w")

class Neural_Network(object):
  def __init__(self):
    self.inputLayerSize = 3 # x1 and x2 and x3
    self.outputLayerSizer = 1 # y1
    self.hiddenLayeSizer = 4
    self.w1 = np.random.randn(self.inputLayerSize,self.hiddenLayeSizer)
    self.w2 = np.random.randn(self.hiddenLayeSizer,self.outputLayerSizer)

  def feedForward(self,x):
    self.z = np.dot(x,self.w1) # prodoct of x input
    self.z2 = self.activationSigmoid(self.z)
    self.z3 = np.dot(self.z2,self.w2)
    o = self.activationSigmoid(self.z3)
    return o

  def backwardPropagate(self , x , y,o):
    self.o_error = y - o # calculate error in output
    self.o_delta = self.o_error * self.activationSigmoidPrime(o)
    self.z2_error = self.o_delta.dot(self.w2.T)
    self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)
    self.w1 +=x.T.dot(self.z2_delta)
    self.w2 += self.z2.T.dot(self.o_delta)

  def trainNetwork(self ,x,y):
    o = self.feedForward(x)
    self.backwardPropagate(x,y,o)

  def activationSigmoid(seld,s):
    return 1/(1+np.exp(-s))

  def activationSigmoidPrime(self,s):
    return s*(1-s)

  def saveSumSquaredLossList(self,i,error):
    lossfile.write(str(i)+","+ str(error.tolist())+ '\n')

  def saveWeight(self):
    np.savetxt("Projects/Machine Learning/XOR Gate/WeightLayer1.txt", self.w1,fmt="%s")
    np.savetxt("Projects/Machine Learning/XOR Gate/WeightLayer2.txt", self.w2,fmt="%s")

  def perdictOutput(self):
    print("Predicted XOR output data based on trained weights:")
    print("Expeceted (X1-X3) : \n" + str(x_predicted))
    print("Output (Y1):\n" + str(self.feedForward(x_predicted)))
      
myNeuralNetwork = Neural_Network()
trainingEpochs = 1000

for i in range(trainingEpochs):
  print("Epochs #" + str(i) + "\n")
  print("Network Input:\n"+str(x))
  print("Expected Output of XOR gate neural network: \n " + str(y))
  print("Actual output fram xor gate neural network: \n" + str(myNeuralNetwork.feedForward(x)))
  loss = np.mean(np.square(y - myNeuralNetwork.feedForward(x)))
  myNeuralNetwork.saveSumSquaredLossList(i,loss)
  print("sum squared loss : \n" + str(loss))
  print("\n")
  myNeuralNetwork.trainNetwork(x,y)

myNeuralNetwork.saveWeight()
myNeuralNetwork.perdictOutput()