import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")
from Data import *
from Data_Fixing_for_RNN import *

def intit_initialVal(sizeOf):
#intialize param
    learn = .001
    nepoch = 50
    T = sizeOf
    hidden = 100
    output = 1
    bptt_truncate = 5
    min_clip_value = -10
    max_clipv_value = 10

    matrixBetweenInputAndHidden = np.random.uniform(0, 1, (hidden, T)) #U
    matrixBetweenHiddenAndOutput = np.random.uniform(0, 1, (output, hidden)) #V
    matrixSharedWeights = np.random.uniform(0, 1, (hidden, hidden)) #W

    return learn, nepoch, T, hidden, bptt_truncate, min_clip_value, max_clipv_value, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights

def ReluFunc(x):
    return 1/ (1 + np.exp(-x))

def cleanX(x):
    #newX = x.flatten()
    x = np.expand_dims(x, axis=2)
    return x

def checkForLossTraining(nepoch, x, y, hidden, T, matrixBetweenInputAndHidden, matrixSharedWeights, matrixBetweenHiddenAndOutput):
    
    tempLoss = 0.0
    for i in range(T):
        newX, newY = x[i], y[i]
        PreviosActivation = np.zeros((hidden, 1))
        for j in range(T):
            newInput = np.zeros(x.shape)
            newInput[j] = newX
            matrixBetweenInputAndHidden2 = np.dot(matrixBetweenInputAndHidden, newInput)
            matrixSharedWeights2 = np.dot(matrixSharedWeights, PreviosActivation)
            tempAdd = matrixBetweenInputAndHidden2 + matrixSharedWeights2
            rel = ReluFunc(tempAdd)
            matrixBetweenHiddenAndOutput2 = np.dot(matrixBetweenHiddenAndOutput, rel)
            PreviosActivation = rel
        
        loss = (x[i] - matrixBetweenHiddenAndOutput2) / 2
        tempLoss += loss
    #print(len(y))
    tempLoss = tempLoss / float(len(y))
    
    
    return matrixBetweenInputAndHidden2, matrixBetweenHiddenAndOutput2, matrixSharedWeights2, tempLoss

def checkLossValidation(x, y, hidden, T, matrixBetweenInputAndHidden, matrixSharedWeights, matrixBetweenHiddenAndOutput):
    tempLoss = 0.0
    for i in range(T):
        newX, newY = x[i], y[i]
        PreviosActivation = np.zeros((hidden, 1))
        for j in range(T):
            newInput = np.zeros(x.shape)
            newInput[j] = newX
            matrixBetweenInputAndHidden2 = np.dot(matrixBetweenInputAndHidden, newInput)
            matrixSharedWeights2 = np.dot(matrixSharedWeights, PreviosActivation)
            tempAdd = matrixBetweenInputAndHidden2 + matrixSharedWeights2
            rel = ReluFunc(tempAdd)
            matrixBetweenHiddenAndOutput2 = np.dot(matrixBetweenHiddenAndOutput, rel)
            PreviosActivation = rel
        
        loss = (x[i] - matrixBetweenHiddenAndOutput2) / 2
        tempLoss += loss
    
    tempLoss = tempLoss / float(len(y))
        
    return tempLoss

def Epoch(epoch, loss, otherloss):
    #print( "Epoch ",x +1, " loss: " tempLoss1, " Val Loss ", temploss2)
    print('Epoch: ', epoch, ', Loss: ', loss, 'Val_Loss ', otherloss)

def training(y, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, T, x, matrixBetweenInputAndHidden2,matrixBetweenHiddenAndOutput2, matrixSharedWeights2):
    for i in range(T):
        layers = []
        newInput = np.zeros((hidden, 1))
        DerivOfMatrixBetweenInputAndHidden = np.zeros(matrixBetweenInputAndHidden.shape)
        DerivOfNatrixBetweenHiddenAndOutput = np.zeros(matrixBetweenHiddenAndOutput.shape)
        DerivOfMatrixSharedWeights = np.zeros(matrixSharedWeights.shape)

        DerivOfMatrixBetweenInputAndHidden_t = np.zeros(matrixBetweenInputAndHidden.shape)
        DerivOfNatrixBetweenHiddenAndOutput_t = np.zeros(matrixBetweenHiddenAndOutput.shape)
        DerivOfMatrixSharedWeights_t = np.zeros(matrixSharedWeights.shape)

        DerivOfMatrixBetweenInputAndHidden_i = np.zeros(matrixBetweenInputAndHidden.shape)
        DerivOfMatrixSharedWeights_i = np.zeros(matrixSharedWeights.shape)

        for t in range(T):
            newInput2 = np.zeros(x.shape)
            newInput2[t] = x[t]
            matrixBetweenInputAndHidden2 = np.dot(matrixBetweenInputAndHidden, newInput2)
            matrixSharedWeights2 = np.dot(matrixSharedWeights, newInput)
            add = matrixBetweenInputAndHidden2 + matrixSharedWeights2
            relu = ReluFunc(add)
            matrixBetweenHiddenAndOutput2 = np.dot(matrixBetweenHiddenAndOutput, relu)
            layers.append({'relu':relu, 'previous Steps':newInput})
            newInput = relu
    return DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, DerivOfMatrixBetweenInputAndHidden_t, DerivOfNatrixBetweenHiddenAndOutput_t, DerivOfMatrixSharedWeights_t, DerivOfMatrixBetweenInputAndHidden_i, DerivOfMatrixSharedWeights_i, layers, newInput, newInput2, relu, matrixBetweenInputAndHidden2, matrixSharedWeights2, add, matrixBetweenHiddenAndOutput2


def backProp(DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput, DerivOfMatrixSharedWeights_t, y, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, T, x, bptt_truncate, layers, add, DerivOfMatrixSharedWeights, DerivOfNatrixBetweenHiddenAndOutput_t, matrixBetweenHiddenAndOutput2, matrixSharedWeights2, matrixBetweenInputAndHidden2, DerivOfMatrixBetweenInputAndHidden_t):
    #DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, DerivOfMatrixBetweenInputAndHidden_t, DerivOfNatrixBetweenHiddenAndOutput_t, DerivOfMatrixSharedWeights_t, DerivOfMatrixBetweenInputAndHidden_i, DerivOfMatrixSharedWeights_i, layers, newInput, newInput2, relu, matrixBetweenInputAndHidden2, matrixSharedWeights2, add, matrixBetweenHiddenAndOutput2 = training(y, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, T, x, nepoch)
    derOfmatrixBetweenHiddenAndOutput3 = (matrixBetweenHiddenAndOutput2 - y.shape)
    for t in range(T):
        DerivOfNatrixBetweenHiddenAndOutput_t = np.dot(derOfmatrixBetweenHiddenAndOutput3, np.transpose(layers[t]['previous Steps']))
        dsv = np.dot(np.transpose(matrixBetweenHiddenAndOutput), derOfmatrixBetweenHiddenAndOutput3)
        ds = dsv
        dadd = add * (1 - add) * ds
        dmulw = dadd * np.ones_like(DerivOfMatrixSharedWeights)
        dprev_s = np.dot(np.transpose(matrixSharedWeights), dmulw)
        for i in range(t-1, max(-1, t-bptt_truncate-1), -1):
            ds = dsv + dprev_s
            dadd = add * (1 - add) * ds
            dmulw = dadd * np.ones_like(matrixSharedWeights2)
            dmulu = dadd * np.ones_like(matrixBetweenInputAndHidden2)
            DerivOfMatrixSharedWeights_i = np.dot(matrixBetweenHiddenAndOutput, layers[t]['previous Steps'])
            dprev_s = np.dot(np.transpose(matrixSharedWeights), dmulw)
            newInput2 = np.zeros(x.shape)
            newInput2[t] = x[t]
            DerivOfMatrixBetweenInputAndHidden_i = np.dot(matrixBetweenInputAndHidden, newInput2)
            dx = np.dot(np.transpose(matrixBetweenInputAndHidden), dmulu)

            DerivOfMatrixBetweenInputAndHidden_t += DerivOfMatrixBetweenInputAndHidden_i
            DerivOfMatrixSharedWeights_t += DerivOfMatrixSharedWeights_i

        DerivOfNatrixBetweenHiddenAndOutput += DerivOfNatrixBetweenHiddenAndOutput_t #v
        DerivOfMatrixBetweenInputAndHidden += DerivOfMatrixBetweenInputAndHidden_t #u
        DerivOfMatrixSharedWeights += DerivOfMatrixSharedWeights_t
    return DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, DerivOfNatrixBetweenHiddenAndOutput_t, DerivOfMatrixSharedWeights_t, DerivOfMatrixBetweenInputAndHidden_i, DerivOfMatrixSharedWeights_i, layers, newInput2, add, dx


def updateWeights(DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, learn, min_clip_value, max_clipv_value):
    if(DerivOfMatrixBetweenInputAndHidden.max() > max_clipv_value):
        DerivOfMatrixBetweenInputAndHidden[DerivOfMatrixBetweenInputAndHidden > max_clipv_value] = max_clipv_value
    if(DerivOfNatrixBetweenHiddenAndOutput.max() > max_clipv_value):
        DerivOfNatrixBetweenHiddenAndOutput[DerivOfNatrixBetweenHiddenAndOutput > max_clipv_value] = max_clipv_value
    if(DerivOfMatrixSharedWeights.max() > max_clipv_value):
        DerivOfMatrixSharedWeights[DerivOfMatrixSharedWeights > max_clipv_value] = max_clipv_value
    
    
    if(DerivOfMatrixBetweenInputAndHidden.min() < min_clip_value):
        DerivOfMatrixBetweenInputAndHidden[DerivOfMatrixBetweenInputAndHidden < min_clip_value] = min_clip_value
    if(DerivOfNatrixBetweenHiddenAndOutput.min() < min_clip_value):
        DerivOfNatrixBetweenHiddenAndOutput[DerivOfNatrixBetweenHiddenAndOutput < min_clip_value] = min_clip_value
    if(DerivOfMatrixSharedWeights.min() < min_clip_value):
        DerivOfMatrixSharedWeights[DerivOfMatrixSharedWeights < min_clip_value] = min_clip_value

    matrixBetweenInputAndHidden -= learn * DerivOfMatrixBetweenInputAndHidden
    matrixBetweenHiddenAndOutput -= learn * DerivOfNatrixBetweenHiddenAndOutput
    matrixSharedWeights -= learn * DerivOfMatrixSharedWeights

    return matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights

def predictions(DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, relu, add, y, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, T, x, MatrixSharedWeights):
    predisesor = []
    for i in range(T):
        
        previous = np.zeros((hidden, 1))
        for t in range(T):
            DerivOfMatrixBetweenInputAndHidden = np.dot(matrixBetweenInputAndHidden, x)
            DerivOfMatrixSharedWeights = np.dot(MatrixSharedWeights, previous)
            add = DerivOfMatrixBetweenInputAndHidden + DerivOfMatrixSharedWeights
            relu = ReluFunc(add)
            DerivOfMatrixBetweenHiddenAndOutput = np.dot(matrixBetweenHiddenAndOutput, relu)
            previous = relu
        predisesor.append(DerivOfMatrixBetweenHiddenAndOutput)
    predisesor = np.array(predisesor)
    return predisesor

def optionPrice( curPrice, strike, riskF = 0.025):
    prices = stockData["Close"][len(stockData)-45:]
    mean = (np.sum(prices) / np.size(prices))
    std = np.std(prices)
    stdPrices = []
    for i in prices:
        diff = i - mean
        std = diff/std
        stdPrices.append(std)
    stdPrices = np.array(stdPrices)
    stdMean = ((np.sum(stdPrices))/np.size(stdPrices))
    stdSize = np.size(stdPrices)
    stdDeviation = []
    for j in stdPrices:
        dif = j - stdMean
        raised = dif**2
        stdDeviation.append(raised)
    stdDeviation = np.array(stdDeviation)
    stdVarriance = (np.sum(stdDeviation)/stdSize)
    vola = math.sqrt(stdVarriance)
    d1 = (math.log(curPrice/strike) + .125*(riskF + (vola**2)/(2)))/(vola * math.sqrt(.125))

    d2 = d1 - vola*math.sqrt(.125)

    normProbd1 = (1/math.sqrt(2*math.pi)) * math.e **-1*((d1**2)/2)
    # May need to figure out if I input d1 as -d1
    normProbd2 = (1/math.sqrt(2*math.pi)) * math.e **-1*((d2**2)/2)
    # May need to figure out if I input d2 as -d2
    optnPrice = (strike*math.e**(-riskF*.125)) * normProbd2 - curPrice*normProbd1
    return optnPrice

def premium():
    curPrice = stockData["Close"].iloc[-1]
    matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, relu, add, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, T = gradient(x_train, y_train, x_test, y_test, 200, sizeOf)
    psudoStrike = curPrice + predictions(DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights,relu, add, y_test, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, newSizeOf,x_test, matrixSharedWeights)[0][0][0]
    premn = optionPrice(curPrice, psudoStrike)
    return premn, psudoStrike

def gradient(x_train, y_train, x_test, y_test, iter, sizeOf):
    learn, nepoch, T, hidden, bptt_truncate, min_clip_value, max_clipv_value, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights = intit_initialVal(sizeOf)
    for i in range(iter):
        matrixBetweenInputAndHidden2, matrixBetweenHiddenAndOutput2, matrixSharedWeights2, tempLoss = checkForLossTraining(nepoch, x_train, y_train, hidden, T, matrixBetweenInputAndHidden, matrixSharedWeights, matrixBetweenHiddenAndOutput)
        tempLossVal = checkLossValidation(x_test, y_test, hidden, T, matrixBetweenInputAndHidden, matrixSharedWeights, matrixBetweenHiddenAndOutput)
        DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, DerivOfMatrixBetweenInputAndHidden_t, DerivOfNatrixBetweenHiddenAndOutput_t, DerivOfMatrixSharedWeights_t, DerivOfMatrixBetweenInputAndHidden_i, DerivOfMatrixSharedWeights_i, layers, newInput, newInput2, relu, matrixBetweenInputAndHidden2, matrixSharedWeights2, add, matrixBetweenHiddenAndOutput2 = training(y_train, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, T, x_train, matrixBetweenInputAndHidden2,matrixBetweenHiddenAndOutput2, matrixSharedWeights2)
        DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, DerivOfMatrixBetweenInputAndHidden_t, DerivOfMatrixSharedWeights_t, DerivOfMatrixBetweenInputAndHidden_i, DerivOfMatrixSharedWeights_i, layers, newInput2, add, dx = backProp(DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput, DerivOfMatrixSharedWeights_t, y_train, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, T, x_train, bptt_truncate, layers, add, DerivOfMatrixSharedWeights, DerivOfNatrixBetweenHiddenAndOutput_t, matrixBetweenHiddenAndOutput2, matrixSharedWeights2, matrixBetweenInputAndHidden2, DerivOfMatrixBetweenInputAndHidden_t)
        matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights = updateWeights(DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, learn, min_clip_value, max_clipv_value)
        #print(A2)
        if i % 5 == 0:
            #print("iteration " + str(i))
            print(Epoch(i, tempLoss,  tempLossVal))
    
    return matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, relu, add, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, T


         


            


            


cleaning = inputs()
stockData = inputs.getStock()



x_train, x_test, y_train, y_test = inputs.getClosing(stockData)
#x_train = (cleanX(x_train))
sizeOf = len(x_train)
newSizeOf = len(x_test)
#matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, matrixSharedWeights, DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights, relu, add, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, T = gradient(x_train, y_train, x_test, y_test, 50, sizeOf)
#arr = predictions(DerivOfMatrixBetweenInputAndHidden, DerivOfNatrixBetweenHiddenAndOutput,DerivOfMatrixSharedWeights,relu, add, y_test, hidden, matrixBetweenInputAndHidden, matrixBetweenHiddenAndOutput, newSizeOf,x_test, matrixSharedWeights)

prem, price = premium()
#premium for a stock at the break even point
print("Premium per contract (contract = 100 shares): " +  str(prem))
#projected price
print("Projected Price per Share: " + str(price))
# print(x_train)
# print("")
