from flask import Flask, request,jsonify, session
import datetime as datedatedate
from datetime import datetime
from datetime import date
from xgboost import XGBRegressor
import xgboost
import pandas_datareader as dr
import tensorflow as tf
from google_trans_new import google_translator

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import requests
from bs4 import BeautifulSoup

from flask_cors import CORS
import math
import yfinance as yf
yf.pdr_override()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os





import dateutil


import pyrebase
import numpy as np
from flask import session

app = Flask(__name__)

CORS(app)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
config = {"apiKey": "AIzaSyCQlJNxtcp_ambo4mGWH9LxhK6Wsr3VlSM",
          "authDomain": "projectbase-1fca0.firebaseapp.com",
          "databaseURL": "https://projectbase-1fca0-default-rtdb.europe-west1.firebasedatabase.app",
          "projectId": "projectbase-1fca0",
          "storageBucket": "projectbase-1fca0.appspot.com",
          "messagingSenderId": "821113244030",
          "appId": "1:821113244030:web:1f86f63dfbba3d08c4cb2f",
          "measurementId": "G-J76JKQ1XX5"}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
auth=firebase.auth()




def handledate(date):
    date=str(date).split(' ')[0]
    return date



@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'GET':
        email = request.args.get('email')
        password = request.args.get('password')
        try:
            user = auth.sign_in_with_email_and_password(email,password)


            dic = dict()
            dic['resp'] = ['done successfully']
            return dic
        except:
            return "problem!"

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'GET':
        email = request.args.get('email')
        password = request.args.get('password')


        try:
            user=auth.create_user_with_email_and_password(email,password)


            dic=dict()
            dic['resp']=['done successfully']
            return dic
        except Exception as e:
            print(str(e))
            return "problem!"



@app.route('/')
def hello_world():
    return 'Hello World!'
#fonction qui fixe la forme de la date from "Jan 26,2020" to "datetime")
def fixdate(date):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    first = date.split(' ')
    for i in range(len(months)):
        if first[0] == months[i]:
            first[0] = str(i + 1)
    first[1] = first[1].replace(',', '')
    date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
    return datetime(date[0], date[1], date[2])


#fonction qui transforme une serie temporelle vers une data supervisé ( x et y )
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values




def comparedate(new):
    date=new.split(' : ')[1]
    date=date.split('/')
    date=datedatedate.date(int('20'+date[2]),int(date[1]),int(date[0]))
    a_month = dateutil.relativedelta.relativedelta(months=4)
    return datedatedate.date.today() - a_month > date

@app.route('/getfrombase', methods=['POST', 'GET'])
def getfrombase():
    if request.method == 'GET':
        dic=db.child('daydata').get().val()
        print('im in from base ----------------------------------------from base')
        return dic

#fonction qui fait la collection et stockage des information des sociétés tunisiens ainsi que les news , l'entrainement des modèles avec les nouvelles données
#input : none
@app.route('/scrap', methods=['POST', 'GET'])
def scrap():
    if request.method == 'GET':

        dic = dict()

        dic['company'] = []
        dic['last'] = []
        dic['chg'] = []
        dic['chgperc'] = []
        dic['date'] = []
        dic['type'] = []
        dic['links'] = []
        dic['symbols'] = []
        dic['news'] = []
        dic['volume'] = []

        try:
            dic = dict()
            dic['company'] = []
            dic['last'] = []

            dic['chgperc'] = []
            dic['date'] = []
            dic['type'] = []
            dic['links'] = []
            dic['symbols'] = []
            dic['news'] = []
            dic['volume'] = []
            dic['high'] = []
            dic['low'] = []
            dic['open'] = []

            url = 'https://www.investing.com/indices/tunindex'
            r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
            soup = BeautifulSoup(r.content, "html.parser")
            body = soup.find('body')
            divtuni = body.find('div', {'data-test': 'instrument-header-details'})

            tuniprice = divtuni.find('span', {'class': 'instrument-price_last__KQzyA'}).text
            print('tuniprice' + str(tuniprice))
            try:
                tunichangeperc = divtuni.find('span', {
                    'class': 'instrument-price_change-percent__19cas instrument-price_down__3dhtw'}).text.replace('(',
                                                                                                                  '').replace(
                    ')', '')
            except:
                tunichangeperc = divtuni.find('span', {
                    'class': 'instrument-price_change-percent__19cas instrument-price_up__2-OcT'}).text.replace('(',
                                                                                                                '').replace(
                    ')', '')
            print('tunichangeperc' + str(tunichangeperc))

            tuniopen = divtuni.find('div', {'class': 'trading-hours_value__2MrOn'}).text
            print('tuniopen' + str(tuniopen))
            tunivolume = '-'
            print('tunivolume' + str(tunivolume))
            try:
                tunichange = divtuni.find('span', {
                    'class': 'instrument-price_change-value__jkuml instrument-price_down__3dhtw'}).text
            except:
                tunichange = divtuni.find('span', {
                    'class': 'instrument-price_change-value__jkuml instrument-price_up__2-OcT'}).text
            print(tunichange)
            for s in body.find('div', class_='gainers-losers-table_table-view-container__Ima20').find_all('td',
                                                                                                          class_='inv-link bold datatable_cell--name__link__1XAxP'):
                dic['type'].append('Gainer')
                dic['company'].append(s.text)

                dic['links'].append(s.find('a', href=True)['href'])

            for s in body.find('div',
                               class_='gainers-losers-table_table-view-container__Ima20 desktop:ml-4 mobileAndTablet:mt-4').find_all(
                    'td',
                    class_='datatable_cell__3gwri datatable_cell--name__CM7yd'):
                dic['company'].append(s.text)
                dic['links'].append(s.find('a', href=True)['href'])
                dic['type'].append('Loser')

            for s in body.find('div',
                               class_='most-active-stocks-table_most-active-stocks-table-container__2Hb6-').find_all(
                    'td',
                    class_='datatable_cell__3gwri datatable_cell--name__CM7yd most-active-stocks-table_second-col__1_cMR'):
                dic['company'].append(s.text)
                dic['links'].append(s.find('a', href=True)['href'])
                dic['type'].append('Most Active')
            toremove = []

            for s in dic['links']:

                url = 'https://www.investing.com' + s

                r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                soup = BeautifulSoup(r.content, "html.parser")
                body = soup.find('body')
                try:
                    h1 = body.find('h1', {
                        'class': 'text-2xl font-semibold instrument-header_title__GTWDv mobile:mb-2'}).text.split(' ')[
                        -1]

                except:
                    try:
                        h1 = body.find('h2', {'class': 'text-lg font-semibold'}).text.split(' ')[0]
                    except:
                        h1 = '-'

                h1 = h1.replace('(', '')
                h1 = h1.replace(')', '')
                dic['symbols'].append(h1)
                # ------------------------------------------endsymbol-----------------------------------------

                url = 'https://www.investing.com' + s + '-historical-data'

                r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                soup = BeautifulSoup(r.content, "html.parser")
                body = soup.find('body')
                try:
                    tds = body.find('table', {'class': 'genTbl closedTbl historicalTbl'}).find_all('td')


                    dic['last'].append(tds[1].text)
                    dic['open'].append(tds[2].text)
                    dic['high'].append(tds[3].text)
                    dic['low'].append(tds[4].text)
                    dic['volume'].append(tds[5].text)
                    dic['chgperc'].append(tds[6].text)
                    dic['date'].append(tds[0].text)
                except:
                    toremove.append(s)
                    tds = body.find('table', {'class': 'genTbl closedTbl historicalTbl'}).find_all('td')
                    dic['date'].append('-')
                    dic['last'].append('-')
                    dic['open'].append('-')
                    dic['high'].append('-')
                    dic['low'].append('-')
                    dic['volume'].append('-')
                    dic['chgperc'].append('-')

            lst = ['type', 'company', 'links', 'symbols', 'date', 'last', 'open', 'high', 'low', 'volume', 'chgperc']
            print(toremove)
            print(len(toremove))
            print('lendic')
            for k in dic.keys():
                print(k)
                print(len(dic[k]))

            if len(toremove) >= 1:
                df = pd.DataFrame()
                for k in dic.keys():
                    if k != 'news':
                        df[k] = dic[k]
                for k in toremove:
                    print(k)
                    df = df[df['links'] != k]
                dic = dict()
                print('i removed bad links')
                for k in df.keys():
                    dic[k] = []
                print('show dataframe len')
                for k in df.keys():
                    print(k)
                    print(len(df[k]))
                df = df.reset_index(drop=True)
                try:
                    for i in range(len(df['links'])):
                        print(str(i) + '---------------------')
                        for k in df.keys():
                            print(k)
                            dic[k].append(df[k][i])
                except Exception as e:
                    print(e)
                dic['news'] = []
                print('im here')

                # -----------------------------------------------------startnews----------------------------------
            ###############################save todays list ################################

            session['todayslist']=dic['symbols']


            ##################################################################################
            for t in dic['symbols']:
                """
                dic['news'].append('-')
                """
                """
                """
                url = 'https://www.ilboursa.com/marches/cotation.aspx?s=' + t
                r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                soup = BeautifulSoup(r.content, "html.parser")
                body = soup.find('body')
                news = []


                try:

                    dv1 = body.find('div', {'class': 'mobpad5'})
                    for new, date in zip(dv1.find_all('a'), dv1.find_all('span', {'class': 'sp1'})):
                        try:

                            news.append([new.text, date.text])


                        except:

                            kkkkk = 15

                    dic['news'].append(news)
                except:

                    dic['news'].append(['no news'])
                """
                """
            print('dic len ---------------------------')
            for k in dic.keys():
                print(len(dic[k]))

            df = pd.DataFrame()
            for key in dic.keys():
                if key != 'links':
                    df[key] = dic[key]
            # ------------------------------------------------------remove double and stock-----------------------------------------------------
            for i in df.index:
                df = df.reset_index(drop=True)
                if i in df.index:
                    for j in df.index:
                        df = df.reset_index(drop=True)

                        if j in df.index:
                            if i != j and df['symbols'][i] == df['symbols'][j]:
                                df['type'][i] = df['type'][i] + '-' + df['type'][j]
                                df = df.drop(df.index[j])


                        else:
                            continue

                else:
                    continue
            for symb in df['symbols']:


                thenewdic = dict()
                print(symb.upper().strip())
                price = df['last'][df['symbols'] == symb].values[0]
                vol = df['volume'][df['symbols'] == symb].values[0]
                change = df['chgperc'][df['symbols'] == symb].values[0]
                date = str(df['date'][df['symbols'] == symb].values[0])

                low = df['low'][df['symbols'] == symb].values[0]
                high = df['high'][df['symbols'] == symb].values[0]
                opn = df['open'][df['symbols'] == symb].values[0]
                thenewdic = db.child('realhistorical').child(symb.upper()).get().val()
                thenewdic['price'].append(str(price))
                thenewdic['date'].append(str(date))
                thenewdic['change'].append(str(change))
                thenewdic['vol'].append(str(vol))
                thenewdic['high'].append(str(high))
                thenewdic['low'].append(str(low))
                thenewdic['open'].append(str(opn))

                new = df['news'][df['symbols'] == symb].values[0]
                try:
                    news = []
                    for n in new:
                        news.append(n[0].replace(',', '').replace(':','') + ' : ' + n[1].replace('\r\n\t\t\t\t\t\t', ''))

                    c = db.child('news').child(symb.upper().strip()).get().val()
                    if str(type(c))!="<class 'NoneType'>":
                        newset=set()
                        for r in c:
                            newset.add(r)
                        for d in news:
                            newset.add(d)
                        lst=[]
                        for e in newset:
                            if comparedate(e)==False:
                                lst.append(e)
                        db.child('news').child(symb.upper().strip()).set(lst)
                    else:
                        lst = []
                        for e in news:
                            if comparedate(e) == False:
                                lst.append(e)

                    db.child('news').child(symb.upper().strip()).set(lst)
                except Exception as e:
                    db.child('news').child(symb.upper().strip()).set(['no news'])

                try:
                    file_name = symb+".pkl"

                    open_file = open('stockmodelsxgboost/' + file_name, "rb")
                    model = pickle.load(open_file)
                    open_file.close()



                    thedic=db.child('realhistorical').child(symb.upper().strip()).get().val()
                    newdf=pd.DataFrame()
                    for k in thedic.keys():
                        newdf[k]=thedic[k]
                    newdf['price'] = newdf['price'].apply(float)

                    lst = []
                    for l in newdf['price']:
                        lst.append(l)

                    newdf['price'] = lst
                    lst2 = []
                    for l in newdf['date']:
                        lst2.append(l)

                    newdf['date'] = lst2
                    newdf['date'] = newdf['date'].apply(fixdate)
                    newdf = newdf.set_index('date')
                    newdf['price'] = newdf['price'].apply(float)
                    newdf = newdf.drop(columns=['change', 'high', 'low', 'open', 'vol'])
                    data = series_to_supervised(newdf[-20:].values, n_in=19)

                    # split dataset

                    # seed history with training dataset
                    history = [x for x in data]
                    d = history[0]
                    d = d.tolist()
                    d.append(float(price))
                    d = np.array(d)
                    history[0] = d

                    print('this is history',history)
                    train = np.asarray(history)
                    print('this is train',train)
                    trainX, trainy = train[:, :-1], train[:, -1]

                    print('this is trainX',trainX)
                    print('this is trainy', trainy)

                    print('training '+symb+' model --------------')


                    model.fit(trainX,trainy)
                    open_file = open('stockmodelsxgboost/' + file_name, "wb")
                    pickle.dump(model, open_file)
                    open_file.close()
                except Exception as e:
                    print(e)









                """
                db.child('realhistorical').child(symb.upper().strip()).set(thenewdic)
                """


            # ----------------------------------------------------------------------------------
            dicdic = dict()
            for k in df.keys():
                dicdic[k] = []
            for i in range(len(df['symbols'])):
                for k in df.keys():
                    dicdic[k].append(str(df[k][i]))

            # return dic
            dicdic['tuniprice'] = []
            dicdic['tuniopen'] = []
            dicdic['tunichangeperc'] = []
            dicdic['tunivolume'] = []
            dicdic['tunichange'] = []
            for i in range(len(dicdic['symbols'])):
                dicdic['tuniprice'].append(tuniprice)
            dicdic['tuniprice'][0]

            for i in range(len(dicdic['symbols'])):
                dicdic['tuniopen'].append(tuniopen)
            dicdic['tuniopen'][0]
            for i in range(len(dicdic['symbols'])):
                dicdic['tunichangeperc'].append(tunichangeperc)
            dicdic['tunichangeperc'][0]
            for i in range(len(dicdic['symbols'])):
                dicdic['tunivolume'].append(tunivolume)
            for i in range(len(dicdic['symbols'])):
                dicdic['tunichange'].append(tunichange)
            print(dicdic['tunichange'][0])
            db.child('daydata').set(dicdic)

            return dicdic
        except Exception as e:
            print('this is scrap error: '+str(e))
            print('except time')
            dic = dict()
            dic['company'] = []
            dic['last'] = []
            dic['chg'] = []
            dic['chgperc'] = []
            dic['closeyes'] = []
            dic['high'] = []
            dic['low'] = []
            dic['open'] = []
            dic['date'] = []
            dic['type'] = []
            dic['links'] = []
            dic['symbols'] = []
            dic['news'] = []
            dic['volume'] = []

            url = 'https://www.ilboursa.com/'
            r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
            soup = BeautifulSoup(r.content, "html.parser")
            body = soup.find('body')
            tds = body.find('div', {'class': 'bar12'}).find_all('td', {'class': 'arr_up'})
            for td in tds:
                dic['symbols'].append(td.find('a').text)
                dic['type'].append('Gainer')

            tds = body.find('div', {'class': 'bar13'}).find_all('td', {'class': 'arr_down'})
            for td in tds:
                dic['symbols'].append(td.find('a').text)
                dic['type'].append('Loser')

            for key in dic['symbols']:
                try:
                    url = 'https://www.ilboursa.com/marches/cotation.aspx?s=' + key
                    r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
                    soup = BeautifulSoup(r.content, "html.parser")
                    body = soup.find('body')
                    last = body.find('div', {'class': 'cot_v1b'}).text.replace('TND', '').strip().replace(',', '.')
                    company = body.find('h1', {'class': 'h1a mob26'}).text
                    try:
                        chgper = body.find('div', {'class': 'quote_up4'}).text
                    except:
                        chgper = body.find('div', {'class': 'quote_down4'}).text

                    date = str(datetime.now()).split(' ')[0]
                    link = '-'
                    symbol = key
                    volume = body.find('div', {'id': 'vol'}).text
                    news = '-'
                    tp = '-'
                    divopenhigh = body.find('div', {'class': 'cot_v21'}).find_all('div')
                    divcloselow = body.find('div', {'class': 'cot_v22'}).find_all('div')
                    closeyes = divcloselow[1].text
                    chg = str((float(chgper.strip().replace('+', '').replace('-', '').replace('%', '').replace(',',
                                                                                                               '.')) / 100) * float(
                    closeyes.strip().replace(',', '.')))
                    high = divopenhigh[3].text
                    low = divcloselow[3].text
                    opn = divopenhigh[1].text
                    dic['company'].append(company)
                    dic['last'].append(last)
                    dic['chg'].append(chg)
                    dic['chgperc'].append(chgper)
                    dic['date'].append(date)

                    dic['links'].append(link)

                    dic['news'].append(news)
                    dic['volume'].append(volume)
                    dic['closeyes'].append(closeyes)
                    dic['high'].append(high)
                    dic['low'].append(low)
                    dic['open'].append(opn)
                    print(key)
                except:
                    print(key)
                    print('error')
            return dic


#output: dictionary contains the keys " price - date - change - high - low - volume - open - symbol
# - tuni index price - tuni index open - tuni index change - tuni index volume "

# dont mind this function
@app.route('/history', methods=['POST', 'GET'])
def history():
    if request.method == 'GET':
        symbol = request.args.get('symbol')
        l = ['company', 'last', 'chg', 'chgperc', 'date', 'type', 'symbols', 'news', 'volume']
        onedict = dict()
        for d in db.child('companies').child(symbol).get().val():
            for i in range(9):

                if l[i] not in onedict.keys():
                    onedict[l[i]] = []
                onedict[l[i]].append(db.child('companies').child(symbol).child(d).get().val()[i][l[i]])

        return onedict


# @app.route('/historical', methods=['POST', 'GET'])
# def historical():
#     if request.method == 'GET':
#         symbol = request.args.get('symbol')
#         dic = db.child('historical').child(symbol).get().val()
#
#         dic['price'].reverse()
#         dic['date'].reverse()
#         dic['high'].reverse()
#         dic['low'].reverse()
#         dic['change'].reverse()
#         dic['open'].reverse()
#         dic['vol'].reverse()
#
#         return dic



#fonction pour recuperer l'historical data / indicateurs financiers /prédiction / news pour une société spécifique
#input : company's symbol
@app.route('/realhistorical', methods=['POST', 'GET'])
def realhistorical():
    if request.method == 'GET':
        symbol = request.args.get('symbol').upper()
        dic = db.child('realhistorical').child(symbol).get().val()

        df = pd.DataFrame()
        for key in dic.keys():
            df[key] = dic[key]
        dfpred=df

        # In[3]:

        df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

        # In[4]:

        def fixdate(date):
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            first = date.split(' ')
            for i in range(len(months)):
                if first[0] == months[i]:
                    first[0] = str(i + 1)
            first[1] = first[1].replace(',', '')
            date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
            return datetime(date[0], date[1], date[2])

        # In[5]:

        df['newdate'] = df['date'].apply(fixdate)
        df['price'] = df['price'].apply(float)

        #------------------------moving average---------------
        dfprice = pd.DataFrame(df.price)
        dfprice['monthly MA'] = dfprice.price.rolling(30).mean()
        dfprice['three MA'] = dfprice.price.rolling(90).mean()
        dfprice['date']=df['newdate']
        dfprice['monthly MA']= dfprice['monthly MA'].apply(str)
        dfprice['three MA'] = dfprice['three MA'].apply(str)
        dic['monthlyMA']=[]
        dic['threeMA'] = []
        for e in dfprice['monthly MA']:
            dic['monthlyMA'].append(e)
        for e in dfprice['three MA']:
            dic['threeMA'].append(e)

        # ------------------------profil---------------
        dicprofil=db.child('profil').get().val()
        dfprofil=pd.DataFrame()
        for k in dicprofil.keys():
            dfprofil[k]=dicprofil[k]
        industry = dfprofil['industry'][dfprofil['symbol'] == symbol].values[0]
        sector = dfprofil['sector'][dfprofil['symbol'] == symbol].values[0]
        story = dfprofil['story'][dfprofil['symbol'] == symbol].values[0]
        dic['industry'] = []
        for i in range(len(dic['price'])):
            dic['industry'].append(str(industry))
        dic['sector'] = []
        for i in range(len(dic['price'])):
            dic['sector'].append(str(sector))
        dic['story'] = []
        for i in range(len(dic['price'])):
            dic['story'].append(str(story))




        #-----------------------------------------------------

        df.set_index('newdate', inplace=True)
        del (df['date'])
        df.sort_index(inplace=True)
        dfpred['price'] = dfpred['price'].apply(float)

        lst = []
        for l in dfpred['price']:
            lst.append(l)

        dfpred['price'] = lst
        lst2 = []
        for l in dfpred['date']:
            lst2.append(l)

        dfpred['date'] = lst2
        dfpred['date'] = dfpred['date'].apply(fixdate)
        dfpred = dfpred.set_index('date')
        dfpred['price'] = dfpred['price'].apply(float)
        dfpred = dfpred.drop(columns=['change', 'high', 'low', 'open', 'vol'])
        data = series_to_supervised(dfpred[-20:].values, n_in=19)

        # split dataset

        # seed history with training dataset
        history = [x for x in data]

        train = np.asarray(history)
        print(train)
        file_name = symbol + ".pkl"


        open_file = open('stockmodelsxgboost/' + file_name, "rb")
        model = pickle.load(open_file)
        open_file.close()




        prediction=str(model.predict(train)[0])

        """
        # In[6]:

        dp = df.loc['2020-5-5':str(df.index[-1])]

        # In[7]:

        # In[8]:

        dp.index = pd.DatetimeIndex(dp.index).to_period('D')

        # In[9]:

        model = sm.tsa.statespace.SARIMAX(dp['price'], order=(0, 1, 0), seasonal_order=(1, 1, 1, 7))
        results = model.fit()

        # In[10]:

        # In[11]:

        # In[12]:

        newdate = dp.index[-1] + timedelta(days=1)
        forecast = results.forecast()[0]

        dic['forecast'] = []
        for i in range(len(dic['price'])):
            dic['forecast'].append(str(forecast))
        """


        dic['forecast'] = []
        for i in range(len(dic['price'])):
            dic['forecast'].append(str(prediction))

        c = db.child('news').child(symbol).get().val()

        if str(type(c)) != "<class 'NoneType'>":
            translator = google_translator()
            sentiment = tf.keras.models.load_model('sentiment/classifier.h5')



            dic['news']=[]
            dic['translated'] = []
            dic['sentiment']=[]
            for e in c:
                dic['news'].append(e)
            #     try:
            #         dic['translated'].append(translator.translate(e, lang_tgt='en'))
            #     except:
            #         dddx=546
            # for e in dic['translated']:
            #     X = tokenizer.texts_to_sequences([e])
            #     X = pad_sequences(X)
            #     model.predict(X)


        df['price']=df['price'].apply(str)
        dic['price']=[]
        for e in df['price']:
            dic['price'].append(e)

        return dic
#output:dictionary with keys "open - high - low - price - date - industry - sector - story - news
# -forecast - monthlyMA - threeMA "


# not now
#get monthly data for specific company
@app.route('/monthly', methods=['POST', 'GET'])
def monthly():
    if request.method == 'GET':
        symbol = request.args.get('symbol')
        dic = db.child('realhistorical').child(symbol).get().val()
        df = pd.DataFrame()
        for key in dic.keys():
            df[key] = dic[key]

        df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

        def fixdate(date):
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            first = date.split(' ')
            for i in range(len(months)):
                if first[0] == months[i]:
                    first[0] = str(i + 1)
            first[1] = first[1].replace(',', '')
            date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
            return datetime(date[0], date[1], date[2])

        df['newdate'] = df['date'].apply(fixdate)
        df['price'] = df['price'].apply(float)
        df.set_index('newdate', inplace=True)
        del (df['date'])
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.resample('1M').mean()

        def handle(x):
            x = str(x).split(' ')[0]
            y = x.split('-')
            x = y[0] + '-' + y[1]
            return x

        df['monthly'] = df.index
        df['monthly'] = df['monthly'].apply(handle)
        df['price'] = df['price'].apply(str)
        dic = dict()
        dic['price'] = []
        dic['monthly'] = []
        for key in df.keys():
            for i in range(len(df[key])):
                dic[key].append(df[key][i])

        return dic

#not now
#get historical data between range
@app.route('/specificdate', methods=['POST', 'GET'])
def specificdate():
    if request.method == 'GET':
        start = request.args.get('start')
        end = request.args.get('end')
        symbol = request.args.get('symbol')

        dic = db.child('realhistorical').child(symbol).get().val()
        df = pd.DataFrame()
        for key in dic.keys():
            df[key] = dic[key]

        df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

        def fixdate(date):
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            first = date.split(' ')
            for i in range(len(months)):
                if first[0] == months[i]:
                    first[0] = str(i + 1)
            first[1] = first[1].replace(',', '')
            date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
            return datetime(date[0], date[1], date[2])

        df['newdate'] = df['date'].apply(fixdate)
        df['price'] = df['price'].apply(float)
        df.set_index('newdate', inplace=True)
        del (df['date'])
        df.sort_index(inplace=True)

        # In[6]:

        dp = df.loc[start:end]
        dp['newdate'] = dp.index
        dp['price'] = dp['price'].apply(str)
        dic = dict()
        for key in dp.keys():
            dic[key] = []

        for key in dp.keys():
            for i in range(len(dp[key])):
                if key == 'newdate':

                    dic[key].append(str(dp[key][i]).split(' ')[0])
                else:
                    dic[key].append(dp[key][i])

        print(dic)

        return dic

#input : company's symbol
#get multiprice data ( for multi price visualization )
@app.route('/listhistorical', methods=['POST', 'GET'])
def listhistorical():
    if request.method == 'GET':
        symbol = request.args.get('symbol')
        alldates = []
        thedictionary = dict()
        symbol = symbol.split(',')
        for symb in symbol:

            dic = db.child('realhistorical').child(str(symb).strip().upper()).get().val()
            df = pd.DataFrame()
            for key in dic.keys():
                df[key] = dic[key]

            # In[3]:

            df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

            # In[4]:

            def fixdate(date):
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                first = date.split(' ')
                for i in range(len(months)):
                    if first[0] == months[i]:
                        first[0] = str(i + 1)
                first[1] = first[1].replace(',', '')
                date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
                return datetime(date[0], date[1], date[2])

            # In[5]:

            df['newdate'] = df['date'].apply(fixdate)
            df['price'] = df['price'].apply(float)
            df.set_index('newdate', inplace=True)
            del (df['date'])
            df.sort_index(inplace=True)
            r = pd.date_range(start=datetime(2013, 1, 1), end=datedatedate.date.today())
            df = df.reindex(r).fillna(np.nan).rename_axis('date').reset_index()
            if len(alldates) == 0:
                alldates = df['date']
            newlst = []
            for s in df['price']:
                if s is not np.nan:
                    newlst.append(str(s))
                else:
                    newlst.append(np.nan)
            df['price'] = newlst
            thedictionary[symb.upper()] = []
            for e in df['price']:
                thedictionary[symb.upper()].append(e)
        thedictionary['date'] = []
        for e in alldates:
            thedictionary['date'].append(str(e).split(' ')[0])

        # df['price']=df['price'].apply(str)
        # dic=dict()
        # dic['date']=[]
        # dic['price']=[]
        # for key in df.keys():
        # for i in df.index:
        # if key=="date":
        # dic[key].append(str(df[key][i]).split(' ')[0])
        # else:
        # dic[key].append(df[key][i])

        return thedictionary
#output : dictionary with keys " date - price "


# not now
#monthly data for multiprice visualization
@app.route('/monthlylisthistorical', methods=['POST', 'GET'])
def monthlylisthistorical():
    if request.method == 'GET':
        symbol = request.args.get('symbol')
        alldates = []
        thedictionary = dict()
        symbol = symbol.split(',')
        for symb in symbol:

            dic = db.child('realhistorical').child(str(symb).strip().upper()).get().val()
            df = pd.DataFrame()
            for key in dic.keys():
                df[key] = dic[key]

            # In[3]:

            df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

            # In[4]:

            def fixdate(date):
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                first = date.split(' ')
                for i in range(len(months)):
                    if first[0] == months[i]:
                        first[0] = str(i + 1)
                first[1] = first[1].replace(',', '')
                date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
                return datetime(date[0], date[1], date[2])

            # In[5]:

            df['newdate'] = df['date'].apply(fixdate)
            df['price'] = df['price'].apply(float)
            df.set_index('newdate', inplace=True)
            del (df['date'])
            df.sort_index(inplace=True)
            r = pd.date_range(start=datetime(2013, 1, 1), end=datedatedate.date.today())
            df = df.reindex(r).fillna(np.nan).rename_axis('date').reset_index()
            df.set_index('date', inplace=True)
            df = df.resample('1M').mean()

            def handle(x):
                x = str(x).split(' ')[0]
                y = x.split('-')
                x = y[0] + '-' + y[1]
                return x

            df['monthly'] = df.index
            df['monthly'] = df['monthly'].apply(handle)
            if len(alldates) == 0:
                alldates = df['monthly']
            newlst = []
            for s in df['price']:
                if s is not np.nan:
                    newlst.append(str(s))
                else:
                    newlst.append(np.nan)
            df['price'] = newlst
            thedictionary[symb.upper()] = []
            for e in df['price']:
                thedictionary[symb.upper()].append(e)
        thedictionary['date'] = []
        for e in alldates:
            thedictionary['date'].append(e)

        # df['price']=df['price'].apply(str)
        # dic=dict()
        # dic['date']=[]
        # dic['price']=[]
        # for key in df.keys():
        # for i in df.index:
        # if key=="date":
        # dic[key].append(str(df[key][i]).split(' ')[0])
        # else:
        # dic[key].append(df[key][i])

        return thedictionary

#input : none
#get historical data for tuni index
@app.route('/historicaltunindex', methods=['POST', 'GET'])
def historicaltunindex():
    if request.method == 'GET':

        dic = db.child('tunindex').get().val()
        df = pd.DataFrame()
        for key in dic.keys():
            df[key] = dic[key]

        df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

        # In[6]:

        df['price'] = df['price'].apply(str)
        dic = dict()
        for key in df.keys():
            dic[key] = []

        for key in df.keys():
            for i in range(len(df[key])):
                dic[key].append(df[key][i])

        print(dic)

        return dic
#output: dictionary with keys " price - date "


#function return the moving average of a time serie
def sma(a,b):
    result = np.zeros(len(a)-b+1)
    for i in range(len(a)-b+1):
        result[i] = np.sum(a[i:i+b])/b
    return result

#function return the exponetial moving average of a time serie
def ema(a,b):
    result = np.zeros(len(a)-b+1)
    result[0] = np.sum(a[0:b])/b
    for i in range(1,len(result)):
        result[i] = result[i-1]+(a[i+b-1]-result[i-1])*(2/(b+1))
    return result


#function return the moving average convergence divergence of a time serie
def macd(a,b,c,d):
    line = ema(a,b)[c-b:]-ema(a,c)
    signal = ema(line,d)
    return line,signal

#input : company's symbol - period
# function that calculate the moving average for the price of a specific company
@app.route('/MAget', methods=['POST', 'GET'])
def MAget():
    if request.method == 'GET':
        symbol = request.args.get('symbol').upper()
        print(symbol)
        period = int(request.args.get('per'))
        print(period)
        dic = db.child('realhistorical').child(symbol).get().val()
        df = pd.DataFrame()
        for k in dic.keys():
            df[k] = dic[k]
        df['price'] = df['price'].apply(float)
        ar = df['price'].to_numpy()
        MA = sma(ar, period)
        lst = []
        for e in MA:
            lst.append(e)
        for e in range(period-1):
            lst.insert(0, np.nan)
        newdic=dict()
        newdic['MA']=[]
        newdic['price'] = []
        newdic['date'] = []
        df['price']=df['price'].apply(str)
        for e in df['price']:
            newdic['price'].append(str(e))


        for e in lst :
            newdic['MA'].append(str(e))

        newdic['date'] = dic['date']
        print(len(newdic['MA']))
        print(len(newdic['price']))
        print(len(newdic['date']))

        return newdic
#output : dictionary with keys " price - moving average - date "

#input : company's symbol - b - c - d
#function calculate macd for the time serie of a specific company
@app.route('/MACDget', methods=['POST', 'GET'])
def MACDget():
    if request.method == 'GET':
        symbol = request.args.get('symbol').upper()
        print(symbol)
        fast = int(request.args.get('fast'))
        slow = int(request.args.get('slow'))
        single = int(request.args.get('single'))

        dic = db.child('realhistorical').child(symbol).get().val()
        df = pd.DataFrame()
        for k in dic.keys():
            df[k] = dic[k]
        df['price'] = df['price'].apply(float)
        ar = df['price'].to_numpy()
        mac = macd(ar,fast,slow,single)
        lst1 = []
        for e in mac[0]:
            lst1.append(e)
        lst2 = []
        for e in mac[1]:
            lst2.append(e)
        for e in range(len(df) - len(lst1)):
            lst1.insert(0, np.nan)
        for e in range(len(df) - len(lst2)):
            lst2.insert(0, np.nan)
        newdic=dict()
        newdic['line']=[]
        newdic['signal'] = []
        newdic['price'] = []
        newdic['date'] = []
        df['price']=df['price'].apply(str)
        for e in df['price']:
            newdic['price'].append(str(e))


        for e in lst1 :
            newdic['line'].append(str(e))
        for e in lst2 :
            newdic['signal'].append(str(e))

        newdic['date'] = dic['date']
        print(len(newdic['line']))
        print(len(newdic['signal']))
        print(len(newdic['price']))
        print(len(newdic['date']))

        return newdic
#output: dictionary contains the keys : " line - signal - price - date " (lists)


#input : company's symbol - period
#function calculate the exponential moving average for the time serie of a specific company
@app.route('/EMAget', methods=['POST', 'GET'])
def EMAget():
    if request.method == 'GET':
        symbol = request.args.get('symbol').upper()
        print(symbol)
        period = int(request.args.get('per'))
        print(period)
        dic = db.child('realhistorical').child(symbol).get().val()
        df = pd.DataFrame()
        for k in dic.keys():
            df[k] = dic[k]
        df['price'] = df['price'].apply(float)
        ar = df['price'].to_numpy()
        EMA = ema(ar, period)
        lst = []
        for e in EMA:
            lst.append(e)
        for e in range(period-1):
            lst.insert(0, np.nan)
        newdic=dict()
        newdic['EMA']=[]
        newdic['price'] = []
        newdic['date'] = []
        df['price']=df['price'].apply(str)
        for e in df['price']:
            newdic['price'].append(str(e))


        for e in lst :
            newdic['EMA'].append(str(e))

        newdic['date'] = dic['date']
        print(len(newdic['EMA']))
        print(len(newdic['price']))
        print(len(newdic['date']))

        return newdic
#output : dictionary with keys " price - exponential moving average - date "

#function that collect the data of a specific company for the last 4 months
#input : company's symbol
@app.route('/getpdf', methods=['POST', 'GET'])
def getpdf():
    if request.method == 'GET':
        symbol = request.args.get('symbol')
        dic = db.child('realhistorical').child(symbol).get().val()
        df = pd.DataFrame()
        for key in dic.keys():
            df[key] = dic[key]

        df = df.drop(columns=['change', 'vol'])

        def fixdate(date):
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            first = date.split(' ')
            for i in range(len(months)):
                if first[0] == months[i]:
                    first[0] = str(i + 1)
            first[1] = first[1].replace(',', '')
            date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
            return datetime(date[0], date[1], date[2])

        df['newdate'] = df['date'].apply(fixdate)
        df['price'] = df['price'].apply(float)
        df['high'] = df['high'].apply(float)
        df['low'] = df['low'].apply(float)
        df['open'] = df['open'].apply(float)
        df.set_index('newdate', inplace=True)
        del (df['date'])
        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)
        dp=df
        a_month = dateutil.relativedelta.relativedelta(months=4)
        startdate=dp.index[-1]-a_month
        dp=dp.loc[startdate:]
        dr=df
        df=df.loc[startdate:]
        dp = dp.resample('1M').mean()

        def handle(x):
            x = str(x).split(' ')[0]
            y = x.split('-')
            x = y[0] + '-' + y[1]
            return x

        df['date'] = df.index
        df['date']=df['date'].apply(str)


        df['bigprice'] = df['price'].apply(str)

        dp['monthly'] = dp.index
        dp['monthly'] = dp['monthly'].apply(handle)
        dp['price'] = dp['price'].apply(str)
        dp['high'] = dp['high'].apply(str)
        dp['low'] = dp['low'].apply(str)
        dp['open'] = dp['open'].apply(str)
        dic = dict()
        dic['price'] = []
        dic['monthly'] = []
        dic['high'] = []
        dic['low'] = []
        dic['open'] = []

        dic['bigprice'] = []
        dic['date'] = []
        for key in dp.keys():
            for i in range(len(dp[key])):
                dic[key].append(dp[key][i])

        for key in ['bigprice','date']:
            if key=="date":

                for i in range(len(df[key])):
                    dic[key].append(str(df[key][i]).split(' ')[0])
            else:
                for i in range(len(df[key])):
                    dic[key].append(df[key][i])



        print(dic)

        #---------------macd
        ar = dr['price'].to_numpy()
        mac = macd(ar, 12, 26, 9)
        lst1 = []
        for e in mac[0]:
            lst1.append(e)
        lst2 = []
        for e in mac[1]:
            lst2.append(e)
        for e in range(len(df) - len(lst1)):
            lst1.insert(0, np.nan)
        for e in range(len(df) - len(lst2)):
            lst2.insert(0, np.nan)

        dic['line'] = []
        dic['signal'] = []

        dic['macddate'] = []
        dr['pricemacd'] = dr['price'].apply(str)
        dic['pricemacd']=[]
        dr['macddate']=dr.index
        for e in dr['macddate']:
            dic['macddate'].append(str(e).split(' ')[0])
        for e in dr['pricemacd']:
            dic['pricemacd'].append(str(e))

        for e in lst1:
            dic['line'].append(str(e))
        for e in lst2:
            dic['signal'].append(str(e))


        return dic
#output: dictionary with the keys : " price - monthly(date) - high - low - open - line - signal
# - macddate - pricemacd "


 #not now
@app.route('/getus', methods=['POST', 'GET'])
def getus():
    if request.method == 'GET':
        url = 'https://www.investing.com/indices/tunindex'
        r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
        soup = BeautifulSoup(r.content, "html.parser")
        body = soup.find('body')
        val = body.find('table', {'class': 'genTbl openTbl quotesSideBlockTbl collapsedTbl'}).find('tr').find_all('td')[2].text
        dic=dict()
        dic['us']=[]
        dic['us'].append(val)
        return dic

#chatbot stuffs
@app.route('/currentprice', methods=['POST', 'GET'])
def currentprice():
    if request.method == 'GET':
        symbol=request.args.get('symbol')
        dic = db.child('realhistorical').child(symbol).get().val()
        currentpr=dic['price'][-1]
        return jsonify(company=symbol,price=currentpr)

#chatbot stuffs
@app.route('/getfinancialreport', methods=['POST', 'GET'])
def financialreport():
    if request.method == 'GET':
        symbol=request.args.get('symbol')
        # dic = db.child('finacialreports').get().val()
        # url=dic[symbol]
        return jsonify(company="val",urlreport="http://www.africau.edu/images/default/sample.pdf")

#chatbot stuffs
@app.route('/getrecommendation', methods=['POST', 'GET'])
def recommend():
    if request.method == 'GET':
        # lst=session['todayslist']
        lst=['ATB','UMED','AL','AB']
        result=dict()
        result['symbol']=[]
        result['prediction']=[]
        result['change'] = []

        def fixdate(date):
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            first = date.split(' ')
            for i in range(len(months)):
                if first[0] == months[i]:
                    first[0] = str(i + 1)
            first[1] = first[1].replace(',', '')
            date = [int(first[2]), int(first[0].replace(' ', '')), int(first[1])]
            return datetime(date[0], date[1], date[2])
        for symbol in lst:
            dic = db.child('realhistorical').child(symbol).get().val()

            df = pd.DataFrame()
            for key in dic.keys():
                df[key] = dic[key]
            dfpred = df

            # In[3]:

            df = df.drop(columns=['change', 'high', 'low', 'open', 'vol'])

            # In[4]:



            # In[5]:

            df['newdate'] = df['date'].apply(fixdate)
            df['price'] = df['price'].apply(float)


            # -----------------------------------------------------

            df.set_index('newdate', inplace=True)
            del (df['date'])
            df.sort_index(inplace=True)
            dfpred['price'] = dfpred['price'].apply(float)

            lst = []
            for l in dfpred['price']:
                lst.append(l)

            dfpred['price'] = lst
            lst2 = []
            for l in dfpred['date']:
                lst2.append(l)

            dfpred['date'] = lst2
            dfpred['date'] = dfpred['date'].apply(fixdate)
            dfpred = dfpred.set_index('date')
            dfpred['price'] = dfpred['price'].apply(float)
            dfpred = dfpred.drop(columns=['change', 'high', 'low', 'open', 'vol'])
            data = series_to_supervised(dfpred[-20:].values, n_in=19)

            # split dataset

            # seed history with training dataset
            history = [x for x in data]

            train = np.asarray(history)
            print(train)
            file_name = symbol + ".pkl"

            open_file = open('stockmodelsxgboost/' + file_name, "rb")
            model = pickle.load(open_file)
            open_file.close()

            prediction = str(model.predict(train)[0])
            result['symbol'].append(symbol)
            result['prediction'].append(prediction)
            result['change'].append(float(prediction)-float(dic['price'][-1]))



        for i in range(len(result['change'])):
            for j in range(i + 1, len(result['change'])):
                if result['change'][i] < result['change'][j]:
                    v = result['change'][i]
                    result['change'][i] = result['change'][j]
                    result['change'][j] = v

                    v = result['symbol'][i]
                    result['symbol'][i] = result['symbol'][j]
                    result['symbol'][j] = v

                    v = result['prediction'][i]
                    result['prediction'][i] = result['prediction'][j]
                    result['prediction'][j] = v

        last=dict()
        last['prediction']=[]
        last['symbol']=[]
        last['change']=[]

        last['prediction']=result['prediction'][:3]
        last['symbol'] = result['symbol'][:3]
        last['change'] = result['change'][:3]

        print(str(last['symbol'][0]))


        return jsonify(symbol=last['symbol'],prediction=last['prediction'],change=last['change'])

#function that collect real time data for crypto currency ( called every 4 second from the frontend)
#input : none
@app.route('/getdata')
def getdata():
    url = 'https://www.investing.com'
    r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
    soup = BeautifulSoup(r.content, "html.parser")
    body = soup.find('body')
    rows = body.find('table', {'class': 'genTbl js-all-crypto-preview-table wideTbl elpTbl elp20 topCryptoHP'}).find(
        'tbody').find_all('tr')
    dic = dict()
    dic['name'] = []
    dic['symbol'] = []
    dic['price'] = []
    dic['market'] = []
    dic['vol'] = []
    dic['chg'] = []
    for row in rows:
        dic['name'].append(row.find_all('td')[1].text)
        dic['symbol'].append(row.find_all('td')[2].text)
        dic['price'].append(row.find_all('td')[3].text)
        dic['market'].append(row.find_all('td')[4].text)
        dic['vol'].append(row.find_all('td')[5].text)
        dic['chg'].append(row.find_all('td')[6].text)

    return dic
#output : dictionary with the keys : "name - price - symbol - market - vol(volume) - chg(change)"

#not now
@app.route('/storedata')
def storedata():
    df = dr.data.get_data_yahoo('ETH-USD', start='2012-01-27')
    df['date'] = df.index
    df['date'] = df['date'].apply(handledate)

    dic = dict()
    for k in df.keys():

        if k not in dic.keys():
            dic[k] = []
        dic[k] = df[k].tolist()

    db.child('cryptodata').child('ETH-USD').set(dic)

    return 'Hello World!'

#not now
@app.route('/dailystore')
def dailystore():
    symbols=['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'ADA-USD']
    for s in symbols:

        model = tf.keras.models.load_model('models/' + s + '.h5')



        df = dr.data.get_data_yahoo(s, start='2012-01-27')

        df['date'] = df.index
        df['date'] = df['date'].apply(handledate)
        dic = dict()
        for k in df.keys():

            if k not in dic.keys():
                dic[k] = []
            dic[k] = df[k].tolist()

        db.child('cryptodata').child(s).set(dic)



        newdf = pd.DataFrame()
        for k in dic.keys():
            newdf[k] = dic[k]
        newdf['Close'] = newdf['Close'].apply(float)
        data = newdf.filter(['Close'])

        dataset = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data = scaler.fit_transform(dataset)
        x_train = []
        y_train = []
        x_train.append(scaled_data[len(scaled_data) - 1 - 60:len(scaled_data) - 1, 0])
        y_train.append(scaled_data[-1, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        model.save('models/' + s + '.h5')
        print('its dooooooooooooooooooooooooooooooooooone')

    return 'done'


#not now
@app.route('/gethistoricalcrypto')
def gethistoricalcrypto():
    if request.method == 'GET':
        symbol = request.args.get('symbol')
        # df = dr.data.get_data_yahoo(symbol, start='2012-06-27')
        #
        dlink = pd.DataFrame()
        dlink['symbol'] = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'ADA-USD']
        dlink['link'] = ['/crypto/bitcoin', '/crypto/ethereum', '/crypto/tether', '/crypto/binance-coin',
                         '/crypto/cardano']

        link=dlink['link'][dlink['symbol'] == symbol].values[0]
        url = 'https://www.investing.com' + link + '/news'
        r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
        soup = BeautifulSoup(r.content, "html.parser")
        body = soup.find('body')
        news = body.find('div', {'class': 'js-articles-wrapper largeTitle js-news-items'}).find_all('div', {
            'class': 'textDiv'})
        #
        #
        #
        #
        # df['date']=df.index
        # df['date']=df['date'].apply(handledate)
        # dic=dict()
        # for k in df.keys():
        #
        #     if k not in dic.keys():
        #         dic[k]=[]
        #     dic[k]=df[k].tolist()
        dic=db.child('cryptodata').child(symbol).get().val()
        df=pd.DataFrame()
        for k in dic.keys():
            df[k]=dic[k]


        # ------------------------moving average---------------
        dfprice = pd.DataFrame(df['Close'])
        dfprice['monthly MA'] = dfprice.Close.rolling(30).mean()
        dfprice['three MA'] = dfprice.Close.rolling(90).mean()

        dfprice['monthly MA'] = dfprice['monthly MA'].apply(str)
        dfprice['three MA'] = dfprice['three MA'].apply(str)


        dic['monthlyMA'] =dfprice['monthly MA'].tolist()
        dic['threeMA'] = dfprice['three MA'].tolist()
        # for e in dfprice['monthly MA']:
        #     dic['monthlyMA'].append(e)
        # for e in dfprice['three MA']:
        #     dic['threeMA'].append(e)
        dic['news']=[]
        dic['news']=[new.find('a').text for new in news]
        #-----------------------------prediction------------------------

        #-----------------------------change mean-----------------------
        # changelst=[]
        # for i in range(len(df['Close'].tolist())):
        #     try:
        #         changelst.append(np.abs(df['Close'].tolist()[i+1]-df['Close'].tolist()[i]))
        #     except:
        #         break
        # changemean=statistics.mean(changelst)


        #---------------------------------------------------------------
        model = tf.keras.models.load_model('models/' + symbol + '.h5')
        data = df.filter(['Close'])

        dataset = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        x_train = []
        x_train.append(scaled_data[len(scaled_data) - 60:len(scaled_data), 0])
        x_train = np.array(x_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        prediction = str(scaler.inverse_transform(model.predict(x_train))[0].tolist()[0])
        prediction=str(prediction)
        lstpred=prediction.split('.')
        try:

            prediction=lstpred[0]+'.'+lstpred[1][0]+lstpred[1][1]
        except:
            try:
                prediction = lstpred[0] + '.' + lstpred[1][0]
            except:
                prediction = lstpred[0]

        realchange=float(prediction)-df['Close'].loc[len(df)-1]

        if realchange>=1:
            message='this stock price is recommended with a positif change of : '+str(realchange)
        else:
            message = 'this stock price is not recommended with a negatif change of : ' + str(realchange)





        dic['prediction']=[prediction]
        dic['message'] = [message]


        return dic

#chatbot stuffs
@app.route('/getstocksummary')
def getstocksummary():
    if request.method == 'GET':

        # start = datetime.strptime(str(date.today()), '%Y-%m-%d')
        #
        # BTC_volume = dr.get_data_yahoo("BTC-USD", start).loc[str(date.today()),'Volume']
        # ETH_volume = dr.get_data_yahoo("ETH-USD", start).loc[str(date.today()), 'Volume']
        # BNB_volume = dr.get_data_yahoo("BNB-USD", start).loc[str(date.today()), 'Volume']
        # USDT_volume = dr.get_data_yahoo("USDT-USD", start).loc[str(date.today()), 'Volume']
        # ADA_volume = dr.get_data_yahoo("ADA-USD", start).loc[str(date.today()), 'Volume']
        # dic=dict()
        # dic['symbol']=['BTC','ETH','BNB','USDT','ADA']
        # dic['volume']=[BTC_volume,ETH_volume,BNB_volume,USDT_volume,ADA_volume]
        # print('im here in stock summary')
        url = 'https://www.investing.com'
        r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
        soup = BeautifulSoup(r.content, "html.parser")
        body = soup.find('body')
        rows = body.find('table',
                         {'class': 'genTbl js-all-crypto-preview-table wideTbl elpTbl elp20 topCryptoHP'}).find(
            'tbody').find_all('tr')
        dic = dict()
        dic['symbol']=['BTC','ETH','BNB','USDT','ADA']
        dic['volume']=[]
        for row in rows:
            print(row.find_all('td')[1].find('a')['href'])
            url = 'https://www.investing.com/' + str(row.find_all('td')[1].find('a')['href']) + '/historical-data'
            r = requests.get(url, headers={"User-Agent": "Opera/9.80"})
            soup = BeautifulSoup(r.content, "html.parser")
            body = soup.find('body')
            rows = body.find('table', {'class': 'genTbl closedTbl historicalTbl'}).find(
                'tbody').find_all('tr')
            volume = rows[0].find_all('td')[5].text
            volume =''.join([i for i in volume if i.isdigit() or i == '.'])
            dic['volume'].append(float(volume))
            print(float(volume))



        return dic


if __name__ == '__main__':
    app.run(threaded=True)