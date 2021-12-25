from json import decoder
import math
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta
import math

class data:
    def __init__(self,ticker):
        self.ticker = ticker

    #generates the equavilent percentile strikes for historical data
    def strikeGeneration(pStrikes, price):
        adjStrikes = np.array([i*price for i in pStrikes])
        return adjStrikes

    #function to Standardize and Normalize a numpy array of data for use in volatility function
    def standardNormal(rawArray, extraArray):
        array = pd.concat([extraArray.tail(4), rawArray])
        mean = np.sum(array) / np.size(array)
        sd = np.std(array)
        normalized  = np.array([((x-mean)/sd) for x in array])
        return normalized

    def volatility(rawArray, extraArray):
        array = data.standardNormal(rawArray["Close"], extraArray["Close"])
        volatility = []
        for index, value in enumerate(array):
            if index > 3:
                prevWeek = array[index-4:index+1]
                mean = (np.sum(prevWeek))/5
                deviation = np.array([(x-mean)**2 for x in prevWeek])
                variance = np.sum(deviation)/5
                currVolatility = math.sqrt(variance)
                volatility.append(currVolatility)
        return volatility
            


    #handles data preprocessing 
    def build(self):
        #print(self.ticker.upper())
        stock = yf.Ticker(self.ticker.upper())
        #closest to 45 days away, initialized as arbitrary days far away
        cl45 = "1999-01-01"
        for i in stock.options:
            if abs(45-(date(int(i[:4]),int(i[5:7]),int(i[8:10]))-date.today()).days) < abs(45-(date(int(cl45[:4]),int(cl45[5:7]),int(cl45[8:10]))-date.today()).days):
                cl45 = i
        #days till expiry
        dte = (date(int(cl45[:4]),int(cl45[5:7]),int(cl45[8:10]))-date.today()).days
        yesterday = str((date.today() - timedelta(days=1)).year) + "-" + str((date.today() - timedelta(days=1)).month) + "-" + str((date.today() - timedelta(days=1)).day)
        option_chain = stock.option_chain(date = cl45).puts
        currentPrice = stock.info['regularMarketPrice']
        strikes = np.array([x for x in option_chain.loc[:,"strike"]])
        percentileStrikes = np.array([i/currentPrice for i in strikes])
        rawHistoricalData = yf.download(tickers = self.ticker, period = "max")
        #Exception Handling for pulling of extra data for companies that have IPO'd after 2010-01-01
        #if '-'.join(i.zfill(2) for i in (str(rawHistoricalData.index[0].to_pydatetime().date().year) + "-" + str(rawHistoricalData.index[0].to_pydatetime().date().month) + "-" + str(rawHistoricalData.index[0].to_pydatetime().date().day)).split('-')) == "2010-01-04":    
        #    extraHistoricalData = yf.download(tickers = self.ticker, start = "2009-12-01", end = "2010-01-01")
        if len(rawHistoricalData) > 750:
            extraStartDate = '-'.join(i.zfill(2) for i in (str(rawHistoricalData.index[0].to_pydatetime().date().year) + "-" + str(rawHistoricalData.index[0].to_pydatetime().date().month) + "-" + str(rawHistoricalData.index[0].to_pydatetime().date().day)).split('-'))
            rawStartDate = extraStartDate[0:5] + str(int(extraStartDate[5:7])+1).zfill(2) + extraStartDate[7:10]
            rawHistoricalData = yf.download(tickers = self.ticker, start = rawStartDate, end = yesterday)
            extraHistoricalData = yf.download(tickers = self.ticker, start = extraStartDate, end = rawStartDate)
        else:
            print("NOT ENOUGH HISTORICAL PRICE DATA")
            return "NOT ENOUGH HISTORICAL PRICE DATA"
        #divider = math.floor(len(rawHistoricalData)*0.7)
        #70% of the data is used for training
        training = rawHistoricalData
        trainingVolatility = data.volatility(training, extraHistoricalData)
        trainingDailyChange = training["Close"] - training["Open"]
        training = training.drop(["High", "Low", "Adj Close"], axis = 1)
        training["Daily Change"] = trainingDailyChange
        training["Volatility"] = trainingVolatility
        #training = training.to_numpy()
        #training = training
        #df = pd.DataFrame(training) #open close volum daily volatil
        return training
    
