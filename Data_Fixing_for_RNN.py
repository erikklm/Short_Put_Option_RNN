import pandas as pd
import numpy as np
from Data import *
from sklearn.model_selection import train_test_split

class inputs:

    def getStock():
        obj = data("MSFT")
        stockData = obj.build()
        #print(stockData.shape)
        num = len(stockData)
        num2 = num-100
        newStockData = stockData[num2:num]
        print(len(newStockData))
        return newStockData


    def getClosing(df):
        df = df[["Close"]]
        print(df.head(4))
        predictInDays = 50
        df["Predeiction"] = df["Close"].shift(-predictInDays)
        arrClosing = np.array(df.drop(["Predeiction"], 1))[:-predictInDays]
        arrPrediction = np.array(df["Predeiction"])[:-predictInDays]


        x_train, x_test, y_train, y_test = train_test_split(arrClosing, arrPrediction, test_size = .50, train_size= .50, random_state=None, shuffle=False, stratify=None)
        # print(x_test.shape)
        # print(y_test.shape)

        return x_train, x_test, y_train, y_test