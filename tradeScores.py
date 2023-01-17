#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import requests
import websocket
import yfinance as yf
import numpy as np
import pandas as pd
from IPython.display import display
import time
import json
import sys
import math
import datetime
from finta import TA
import pytz
format = "%Y-%m-%d %H:%M:%S"
formatters = {
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[33m',
    'END': '\033[0m',
}
import talib
import slack
import xlwings as xw
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
import multiprocessing as mp
format = "%Y-%m-%d %H:%M:%S"
#Class definition
class MultiProcessingFunctions:
	""" This static functions in this class enable multi-processing"""
	def __init__(self):
		pass

	@staticmethod

	def lin_parts(num_atoms, num_threads):
		""" This function partitions a list of atoms in subsets (molecules) of equal size.
		An atom is a set of indivisible set of tasks.
		"""

		# partition of atoms with a single loop
		parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
		parts = np.ceil(parts).astype(int)
		return parts

	@staticmethod
	def nested_parts(num_atoms, num_threads, upper_triangle=False):
		""" This function enables parallelization of nested loops.
		"""
		# partition of atoms with an inner loop
		parts = []
		num_threads_ = min(num_threads, num_atoms)

		for num in range(num_threads_):
			part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.) / num_threads_)
			part = (-1 + part ** .5) / 2.
			parts.append(part)

		parts = np.round(parts).astype(int)

		if upper_triangle:  # the first rows are heaviest
			parts = np.cumsum(np.diff(parts)[::-1])
			parts = np.append(np.array([0]), parts)
		return parts

	@staticmethod
	def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, **kargs):
		"""	
		:param func: (string) function to be parallelized
		:param pd_obj: (vector) Element 0, is name of argument used to pass the molecule;
						Element 1, is the list of atoms to be grouped into a molecule
		:param num_threads: (int) number of threads
		:param mp_batches: (int) number of batches
		:param lin_mols: (bool) Tells if the method should use linear or nested partitioning
		:param kargs: (var args)
		:return: (data frame) of results
		"""

		if lin_mols:
			parts = MultiProcessingFunctions.lin_parts(len(pd_obj[1]), num_threads * mp_batches)
		else:
			parts = MultiProcessingFunctions.nested_parts(len(pd_obj[1]), num_threads * mp_batches)

		jobs = []
		for i in range(1, len(parts)):
			job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
			job.update(kargs)
			jobs.append(job)

		if num_threads == 1:
			out = MultiProcessingFunctions.process_jobs_(jobs)
		else:
			out = MultiProcessingFunctions.process_jobs(jobs, num_threads=num_threads)

		if isinstance(out[0], pd.DataFrame):
			df0 = pd.DataFrame()
		elif isinstance(out[0], pd.Series):
			df0 = pd.Series()
		else:
			return out

		for i in out:
			df0 = df0.append(i)

		df0 = df0.sort_index()
		return df0

	@staticmethod
	def process_jobs_(jobs):
		""" Run jobs sequentially, for debugging """
		out = []
		for job in jobs:
			out_ = MultiProcessingFunctions.expand_call(job)
			out.append(out_)
		return out

	@staticmethod
	def expand_call(kargs):
		""" Expand the arguments of a callback function, kargs['func'] """
		func = kargs['func']
		del kargs['func']
		out = func(**kargs)
		return out

	@staticmethod
	def report_progress(job_num, num_jobs, time0, task):
		# Report progress as asynch jobs are completed

		msg = [float(job_num) / num_jobs, (time.time() - time0)/60.]
		msg.append(msg[1] * (1/msg[0] - 1))
		time_stamp = str(dt.datetime.fromtimestamp(time.time()))

		msg = time_stamp + ' ' + str(round(msg[0]*100, 2)) + '% '+task+' done after ' + 			str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

		if job_num < num_jobs:
			sys.stderr.write(msg+'\r')
		else:
			sys.stderr.write(msg+'\n')

		return

	@staticmethod
	def process_jobs(jobs, task=None, num_threads=24):
		""" Run in parallel. jobs must contain a 'func' callback, for expand_call"""

		if task is None:
			task = jobs[0]['func'].__name__

		pool = mp.Pool(processes=num_threads)
		# outputs, out, time0 = pool.imap_unordered(MultiProcessingFunctions.expand_call,jobs),[],time.time()
		outputs = pool.imap_unordered(MultiProcessingFunctions.expand_call, jobs)
		out = []
		time0 = time.time()

		# Process asyn output, report progress
		for i, out_ in enumerate(outputs, 1):
			out.append(out_)
			MultiProcessingFunctions.report_progress(i, len(jobs), time0, task)

		pool.close()
		pool.join()  # this is needed to prevent memory leaks
		return out

#Volatility:
# def getDailyVol(close,span0=100):
#     # daily vol, reindexed to close 
#     df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
#     df0=df0[df0>0] 
#     df0=pd.Series(close.index[df0 - 1], index=close.index[close.shape[0]-df0.shape[0]:]) 
#     df0=close.loc[df0.index]/close.loc[df0.values].values-1 
#     # daily returns 
#     df0=df0.ewm(span=span0).std() 
#     return df0
#Tick selection:
def getTEvents(close,h):
    tEvents,sPos,sNeg=[],0,0 
    diff = np.log(close).diff().dropna() 
    for i in diff.index[1:]:
        sPos=max(0,(sPos+diff.loc[i]))
        sNeg=min(0,(sNeg+diff.loc[i]))
        if sNeg<-h:
            sNeg=0;tEvents.append(i) 
        elif sPos>h:
            sPos=0;tEvents.append(i) 
    return pd.DatetimeIndex(tEvents)
#Triple Barrier:
def applyPtSlOnT1(close,events,ptSl,molecule):
# apply stop loss/profit taking, if it takes place before t1 (end of event) 
    events_=events.loc[molecule] 
    out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0:
        pt=ptSl[0]*events_['trgt'] 
    else:pt=pd.Series(index=events.index) # NaNs 
    if ptSl[1]>0:
        sl=-ptSl[1]*events_['trgt']
    else:sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_['t1'].ﬁllna(close.index[-1]).iteritems():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    return out
# LIst of events:
def getEvents(close,tEvents,ptSl,trgt,minRet,numThreads,t1=False,side=None):
#1) get target 
    trgt=trgt.loc[tEvents] 
    trgt=trgt[trgt>minRet] # minRet 
    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT,index=tEvents) 
        #3) form events object, apply stop loss on t1 
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index),[ptSl[0],ptSl[0]] 
    else:side_,ptSl_=side.loc[trgt.index],ptSl[:2] 
    events=pd.concat( { 't1':t1,'trgt':trgt,'side':side_ } ,axis=1).dropna(subset=['trgt']) 
    df0=MultiProcessingFunctions.mp_pandas_obj(func=applyPtSlOnT1,pd_obj=('molecule',events.index),
                                               num_threads=numThreads,close=close,events=events,ptSl=ptSl_)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan 
    if side is None:events=events.drop('side',axis=1)
    return events
# Add Vertical Barriers:
def add_vertical_barrier(t_events, close, num_days=1):
    """
    :param t_events: (series) series of events (symmetric CUSUM filter)
    :param close: (series) close prices
    :param num_days: (int) maximum number of days a trade can be active
    :return: (series) timestamps of vertical barriers
    """
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])  # NaNs at end
    return t1
# Barrier Touched:
def barrier_touched(out_df):
    """
    :param out_df: (DataFrame) containing the returns and target
    :return: (DataFrame) containing returns, target, and labels
    """
    store = []
    for i in np.arange(len(out_df)):
        date_time = out_df.index[i]
        ret = out_df.loc[date_time, 'ret']
        target = out_df.loc[date_time, 'trgt']

        if ret > 0.0 and ret > target:
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and ret < -target:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)
    out_df['bin'] = store
    return out_df
#Get Bins:
def getBins(events,close):
    ''' Compute event's outcome (including side information, if provided). events is a DataFrame where:
    —events.index is event's starttime —events[’t1’] is event's endtime —events[’trgt’] 
    is event's target —events[’side’] (optional) implies the algo's position side Case 1: 
    (’side’ not in events): bin in (-1,1) <—label by price action Case 2: (’side’ in events):
    bin in (0,1) <—label by pnl (meta-labeling)
    '''
    #1) prices aligned with events 
    events_=events.dropna(subset=['t1']) 
    px=events_.index.union(events_['t1'].values).drop_duplicates() 
    px=close.reindex(px,method='bfill') 
    #2) create out object 
    out=pd.DataFrame(index=events_.index) 
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1 
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling 
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out


##########################################################################################################################
def tech_score(j,start_dt,end_dt,interval):
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
        for i in range(5):
            if((adx[-i]>25)
            and (dmi['DI+'][-i-1]<dmi['DI-'][-i-1])
            and (ma['MACD'][-i] >ma['MACD'][-i-1]) 
            and (ma['MACD'][-i] >ma['SIGNAL'][-i]) 
            and (ohlc['close'][-i] >ohlc['open'][-i])
            and (ma['MACD'][-i-1] <=ma['SIGNAL'][-i-1])
            and (ma['MACD'][-i-2] <=ma['SIGNAL'][-i-2])
            #and ((ohlc['high'][-i]-ohlc['close'][-i])<(ohlc['open'][-i]-ohlc['low'][-i]))
            and (rsi[-i-1]<30)):
              buy_picks.setdefault(j,[]).append((ohlc.index[-i]).strftime('%H:%M'))
              buy_picks.setdefault(j,[]).append("BUY")
              buy_picks.setdefault(j,[]).append(round(ohlc['close'][-i],2))
            elif((adx[-i]>25)
            and (dmi['DI+'][-i-1] >dmi['DI-'][-i-1])
            and (ma['MACD'][-i] <ma['MACD'][-i-1]) 
            and (ma['MACD'][-i] <ma['SIGNAL'][-i]) 
            and (ohlc['close'][-i] <ohlc['open'][-i])
            and (ma['MACD'][-i-1] >=ma['SIGNAL'][-i-1])
            and (ma['MACD'][-i-2] >=ma['SIGNAL'][-i-2])):
            #and ((ohlc['high'][-i]-ohlc['close'][-i])>(ohlc['open'][-i]-ohlc['low'][-i]))
            #and (rsi[-i-1]>70)):
              buy_picks.setdefault(j,[]).append((ohlc.index[-i]).strftime('%H:%M'))
              #s="{RED}SELL{END}!".format(**formatters)
              buy_picks.setdefault(j,[]).append("SELL")
              buy_picks.setdefault(j,[]).append(round(ohlc['close'][-i],2))
        if(len(buy_picks.get(j))<=1):
            buy_picks.setdefault(j,[]).append((ohlc.index[-1]).strftime('%H:%M'))
            buy_picks.setdefault(j,[]).append("HOLD")
            buy_picks.setdefault(j,[]).append(round(ohlc['close'][-1],2))
        for k in buy_picks.get(j):
          if(k=='BUY'):
            dec=1
            print('{GREEN}BUY{END}'.format(**formatters), end =" ")
          elif(k=='SELL'):
            dec=0
            print('{RED}SELL{END}'.format(**formatters), end =" ")
          elif(k=='HOLD'):
            dec=2
            print('{YELLOW}HOLD{END}'.format(**formatters), end =" ")
          else:
            print(k, end =" ")
    return dec
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
        print(j,ohlc.index[-1],end=" ")
        if(b=='BUY'):
            dec=1
            print('{GREEN}BUY{END}'.format(**formatters), end =" ")
        elif(b=='SELL'):
            dec=0
            print('{RED}SELL{END}'.format(**formatters), end =" ")
        elif(b=='HOLD'):
            dec=2
            print('{YELLOW}HOLD{END}'.format(**formatters), end =" ")
        print(round(ohlc['close'][-i],2),"Accuracy: ",acc)
    return order[0]