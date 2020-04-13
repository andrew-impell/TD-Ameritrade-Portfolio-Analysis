import pandas as pd
import pandas_datareader as pdr
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)

ticker = str(input("Enter a ticker to analyse: "))

df = pd.read_csv('Stocks_raw_data.csv', header=0)

tickers = list(df['SYMBOL'].unique())
tickers.pop(0)


df_ticker = df[df['SYMBOL'] == ticker]
df_ticker['IsSold'] = df_ticker['DESCRIPTION'].str.contains('Sold')


def graph_buy_sell(df_ticker):
    sold_dates = []
    buy_dates = []

    df_ticker.drop(df_ticker.tail(1).index, inplace=True)

    df_ticker['DATE'] = pd.to_datetime(df_ticker['DATE'],  format='%m/%d/%Y')

    df_ticker['Sold DATEs'] = \
        df_ticker['DATE'][df_ticker['IsSold'] == True]
    sell_list = list(df_ticker['Sold DATEs'].dropna().values)
    df_ticker['Buy DATEs'] =\
        df_ticker['DATE'][df_ticker['IsSold'] == False]
    buy_list = list(df_ticker['Buy DATEs'].dropna().values)

    all_dates = sell_list + buy_list
    last_date = max(all_dates)
    first_date = min(all_dates)

    last = str(last_date)[:10]
    first = str(first_date)[:10]
    tick_df = pdr.get_data_yahoo(ticker, first, last)
    price = tick_df['Adj Close'].round(5)

    plt.plot(price.index, price)
    plt.scatter(buy_list, price.loc[buy_list], label='Buy', color='r')
    plt.scatter(sell_list, price.loc[sell_list], label='Sell', color='g')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker}')
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()


graph_buy_sell(df_ticker)
