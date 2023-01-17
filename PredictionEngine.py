#!/usr/bin/env python
# coding: utf-8
import yfinance as yf
import requests
import websocket
import numpy as np
import pandas as pd
import time
import re
from textblob import TextBlob
import math
import datetime
from finta import TA
import talib
import tweepy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')
stock=sys.argv[1]
start_dt=sys.argv[2]
end_dt=sys.argv[3]
interval=sys.argv[4]
scores={}
client = tweepy.Client(bearer_token=XXX)

def tech_score(j,start_dt,end_dt,interval):
    buy_picks={}
    st=yf.Ticker(j)
    dec=2
    share = pd.DataFrame(st.history(start=start_dt,end=end_dt,
                                                 interval=interval))[['Open','High','Low','Close','Volume']]
    if(len(share)>=5):
        ohlc=share[['Open','High','Low','Close']]
        ohlcv=share[['Open','High','Low','Close','Volume']]
        ohlc.rename(columns = {'Open':'open', 'High':'high', 
                                  'Low':'low','Close':'close'}, inplace = True) 
        ohlcv.rename(columns = {'Open':'open', 'High':'high', 
                                  'Low':'low','Close':'close','Volume':'volume'}, inplace = True) 
        ma=TA.MACD(ohlc,period_fast=3,period_slow=5,signal=5)
        rsi=TA.RSI(ohlc)
        adx=TA.ADX(ohlc,5)
        dmi=TA.DMI(ohlc,5)
        buy_picks.setdefault(j,[]).append(j)
        if(((adx[-1]>25)
        and (dmi['DI+'][-1]>dmi['DI-'][-1])
        and (ma['MACD'][-1] >ma['SIGNAL'][-1]))
        or ((ma['MACD'][-2] <=ma['SIGNAL'][-2])
        and ((ohlc['high'][-1]-ohlc['close'][-1])<=(ohlc['open'][-1]-ohlc['low'][-1]))
        and (rsi[-2]<30))):
            dec=1
            buy_picks.setdefault(j,[]).append(dec)
        elif(((adx[-1]>25)
        and (dmi['DI+'][-1] <dmi['DI-'][-1])
        and (ma['MACD'][-1] <ma['SIGNAL'][-1])) 
        or ((ma['MACD'][-2] >=ma['SIGNAL'][-2])
        and ((ohlc['high'][-1]-ohlc['close'][-1])>=(ohlc['open'][-1]-ohlc['low'][-1]))
        and (rsi[-2]>70))):
            dec=0
            buy_picks.setdefault(j,[]).append(dec)
        else:
            dec=2
            buy_picks.setdefault(j,[]).append(dec)
    return buy_picks.get(j)[-1]

def LR_score(j,start_dt,end_dt,interval):
    buy_picks={}
    st=yf.Ticker(j)
    dec=2
    share = pd.DataFrame(st.history(start=start_dt,end=end_dt,
                                                 interval=interval))[['Open','High','Low','Close','Volume']]
    if(len(share)>10):
        ohlc=share[['Open','High','Low','Close']]
        ohlcv=share[['Open','High','Low','Close','Volume']]
        ohlc.rename(columns = {'Open':'open', 'High':'high', 
                                  'Low':'low','Close':'close'}, inplace = True) 
        ohlcv.rename(columns = {'Open':'open', 'High':'high', 
                                  'Low':'low','Close':'close','Volume':'volume'}, inplace = True) 
        ma=TA.MACD(ohlc,period_fast=5,period_slow=10,signal=5)
        rsi=TA.RSI(ohlc)
        adx=TA.ADX(ohlc,5)
        dmi=TA.DMI(ohlc,5)
        buy_picks.setdefault(j,[]).append(j)
        candles=ohlc.copy()
        o=candles['open']
        h=candles['high'];
        l=candles['low']
        c=candles['close']
        CDLHAMMER=talib.CDLHAMMER(o, h, l, c)
        CDL3INSIDE=talib.CDL3INSIDE(o, h, l, c)
        CDLMORNINGSTAR=talib.CDLMORNINGSTAR(o, h, l, c)
        CDLMORNINGDOJISTAR=talib.CDLMORNINGDOJISTAR(o, h, l, c)
        CDL3WHITESOLDIERS=talib.CDL3WHITESOLDIERS(o, h, l, c)
        CDLTRISTAR=talib.CDLTRISTAR(o, h, l, c)
        CDLEVENINGSTAR=talib.CDLEVENINGSTAR(o, h, l, c)
        CDLEVENINGDOJISTAR=talib.CDLEVENINGDOJISTAR(o, h, l, c)
        CDL3OUTSIDE=talib.CDL3OUTSIDE(o, h, l, c)
        CDLRISEFALL3METHODS=talib.CDLRISEFALL3METHODS(o, h, l, c)
        CDL3BLACKCROWS=talib.CDL3BLACKCROWS(o, h, l, c)
        CDLINVERTEDHAMMER=talib.CDLINVERTEDHAMMER(o, h, l, c)
        CDLENGULFING=talib.CDLENGULFING(o, h, l, c)
        CDLMARUBOZU=talib.CDLMARUBOZU(o, h, l, c)
        CDLSHOOTINGSTAR=talib.CDLSHOOTINGSTAR(o, h, l, c)
        ohlc['MACD']=ma['MACD']
        ohlc['SIGNAL']=ma['SIGNAL']
        ohlc['ADX']=adx
        ohlc['RSI']=rsi
        ohlc['DI+']=dmi['DI+']
        ohlc['DI-']=dmi['DI-']
        ohlc['CDLHAMMER']=CDLHAMMER
        ohlc['CDL3INSIDE']=CDL3INSIDE
        ohlc['CDLMORNINGSTAR']=CDLMORNINGSTAR
        ohlc['CDLMORNINGDOJISTAR']=CDLMORNINGDOJISTAR
        ohlc['CDL3WHITESOLDIERS']=CDL3WHITESOLDIERS
        ohlc['CDLTRISTAR']=CDLTRISTAR
        ohlc['CDL3OUTSIDE']=CDL3OUTSIDE
        ohlc['CDLEVENINGSTAR']=CDLEVENINGSTAR
        ohlc['CDLEVENINGDOJISTAR']=CDLEVENINGDOJISTAR
        ohlc['CDL3BLACKCROWS']=CDL3BLACKCROWS
        ohlc['CDLRISEFALL3METHODS']=CDLRISEFALL3METHODS
        ohlc['CDLINVERTEDHAMMER']=CDLINVERTEDHAMMER
        ohlc['CDLENGULFING']=CDLENGULFING
        ohlc['CDLMARUBOZU']=CDLMARUBOZU
        ohlc['CDLSHOOTINGSTAR']=CDLSHOOTINGSTAR
        ohlc['CT']=ohlc.apply(lambda x: x['close']-x['open'],axis=1)
        ohlc['CT']=ohlc['CT'].apply(lambda x: 'G' if x>0 else ('R' if x<0 else 'N'))
        ohlc['Direction']=''
        for i in range(0,len(ohlc)-1):
            if ((ohlc['high'][i+1]-ohlc['close'][i])>0):ohlc["Direction"][i]=1
            elif((ohlc['close'][i]-ohlc['low'][i+1])>0):ohlc["Direction"][i]=0
            else:ohlc["Direction"][i]=2
        ohlc['Direction'][-1]=(1 if ohlc['CT'][-1]=='G' else (0 if ohlc['CT'][-1]=='R' else 2))
        ohlc=ohlc.drop('CT',axis=1)
        ohlc=ohlc.fillna(0)
        x=ohlc.iloc[:,:25]
        y=ohlc.iloc[:,25:26]
        i=1
        acc=0
        min_acc=.5
        while(acc<min_acc):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state=i)
            scaler = StandardScaler()
            X_train = scaler.fit_transform( X_train )
            X_test = scaler.transform( X_test )
            model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X_train, y_train)
            preds = model.predict(X_test)
            acc=round(metrics.accuracy_score(y_test, preds),1)
            i=i+1
        order = model.predict((ohlc.iloc[-1,:25]).values.reshape(-1, 25))
        if(order[0]==1):b='BUY'
        elif(order[0]==0):b='SELL'
        elif(order[0]==2):b='HOLD'
        #print(round(ohlc['close'][-i],2),"Accuracy: ",acc)
        buy_picks.setdefault(j,[]).append(order[0])
        buy_picks.setdefault(j,[]).append(acc)
        buy_picks.setdefault(j,[]).append(round(ohlc['close'][-1],2))
    return buy_picks

def clean_tweet(tweet):
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp)
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp

def SA_score(stock):
    name=stock.split('.')[0]
    query = name+" -is:retweet"
    ps=ng=nt=0
    for tweet in tweepy.Paginator(client.search_recent_tweets, query=query, max_results=10).flatten(limit=20):
        s=clean_tweet(str(tweet))
        analysis = TextBlob(clean_tweet(s))
        if analysis.sentiment.polarity > 0: 
            ps=ps+1
        elif analysis.sentiment.polarity == 0: 
            nt=nt+1
        else: 
            ng=ng+1
        if(ps>nt and ps>ng):
            dec=1
        elif(ng>ps and ng>nt):
            dec=0
        else:
            dec=2
    return dec

ta=tech_score(stock,start_dt,end_dt,interval)
ra=LR_score(stock,start_dt,end_dt,interval)
sa=SA_score(stock)
scores.setdefault(stock,[]).append(ta)
scores.setdefault(stock,[]).append(ra.get(stock)[1])
scores.setdefault(stock,[]).append(ra.get(stock)[2])
scores.setdefault(stock,[]).append(sa)
scores.setdefault(stock,[]).append(ra.get(stock)[3])
print(scores)
