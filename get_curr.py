import pandas as pd
import requests
from datetime import date
from pandas.io.json import json_normalize



today = date.today()


def construct_url(ticker, token, startDate=today):
    first = "https://api.tiingo.com/tiingo/daily/" + str(ticker.lower())
    third = "/prices?startDate=" + str(startDate)
    last = "&token=" + str(token)

    url = first + third + last

    return url


print(construct_url('AAPL', tiingo_api))


def get_price(ticker, token):
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        requestResponse = requests.get(
            construct_url(ticker, token), headers=headers)
        req = requestResponse.json()

        df_json = json_normalize(req)

        price = df_json['open'][0]

        return price
    except:
        print(f"Cannot return ticker {ticker}")


df = pd.read_csv('Stocks_raw_data.csv', header=0)

df_out_list = []

tickers = list(df['SYMBOL'].unique())
tickers.pop(0)

print(tickers)

for ticker in tickers:
    price = get_price(ticker, tiingo_api)
    print(price, ticker)
