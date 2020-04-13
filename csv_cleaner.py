import requests
import pandas as pd
import numpy as np
import datetime
# from pandas.io.json import json_normalize
import os
from tqdm import tqdm
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('fivethirtyeight')
pd.set_option('mode.chained_assignment', None)

tiingo_api = ''

grab_current_price = True
output_file = True

out_filename = 'portfolio_stats'

today = datetime.date.today()


def get_open_close():
    dt = datetime.datetime.now()
    if dt.hour < 17:
        return 'Open'
    else:
        return 'Adj Close'


def get_price_test(ticker, df_ticker):
    mu = df_ticker['PRICE'].mean()
    sigma = df_ticker['PRICE'].std()
    return np.abs(np.random.normal(mu, sigma))


def get_neg(s):
    '''
    Transforms True false column to -1 or 1 to multiply
    '''
    if s:
        return -1
    else:
        return 1


def construct_url(ticker, token, startDate=today):
    '''
    Creates a URL for the Tiingo API to request a price from
    :params
    ticker: which ticker to grab
    token: api token

    :returns
    str: URL

    '''
    first = "https://api.tiingo.com/tiingo/daily/" + str(ticker.lower())
    third = "/prices?startDate=" + str(startDate)
    last = "&token=" + str(token)

    url = first + third + last

    return url


def get_price(ticker, token, df_ticker):
    '''
    try to get a close price on every ticker
    :params
    ticker
    token
    :returns
    float price
    '''
    '''
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        requestResponse = requests.get(
            construct_url(ticker, token), headers=headers)
        req = requestResponse.json()

        df_json = json_normalize(req)

        price = df_json['close'][0]

        return price
    '''
    try:
        tick_df = pdr.get_data_yahoo(ticker)
        price = tick_df[get_open_close()].iloc[-1].round(5)

        return price

    except Exception as e:
        print(e)
        print(f"Cannot return ticker {ticker}")
        return np.abs(np.random.normal(df_ticker['PRICE'].mean(),
                                       df_ticker['PRICE'].std()))


def get_normal(ser):

    lis = np.array(ser.values)
    lis.sort()

    mu = np.mean(lis)
    sigma = np.std(lis)

    y = norm.pdf(lis, mu, sigma)
    return lis, y


def graph_hists(df):
    fig, (ax1, ax2) = plt.subplots(2)

    GL = df['Gain/Loss']

    GL = GL[~((GL-GL.mean()).abs() > 3 * GL.std())]

    x, y = get_normal(df['Gain/Loss'])
    x2, y2 = get_normal(df['Total Dividend'])

    mu = df['Gain/Loss'].mean()

    ax1.hist(df['Gain/Loss'], density=True, bins=200, alpha=0.6)
    ax1.set_xlim([-150, 150])
    ax1.set_ylim([0, 0.025])
    ax1.vlines(df['Gain/Loss'].mean(), ymin=0, ymax=0.06, linestyle='dashed',
               label=f'Mean={mu:.4}', linewidth=1)
    ax1.legend()
    ax1.plot(x, y)

    ax1.set_title('Gain/Loss')
    ax1.set_xlabel('Gain/Loss')
    ax1.set_ylabel('Freq.')

    ax2.hist(df['Total Dividend'], alpha=0.6)

    ax2.plot(x2, y2)
    ax2.set_title('Total Dividends')
    ax2.set_xlabel('Dividend')
    ax2.set_ylabel('Freq.')

    plt.subplots_adjust(wspace=0.2, hspace=0.7)

    plt.show()


# get dataframe
df = pd.read_csv('Stocks_raw_data.csv', header=0)

df_div = df.copy()
df_div = df_div[df_div['DESCRIPTION'].notnull()]
df_dividend = df_div[df_div['DESCRIPTION'].str.contains("DIVIDEND")]

df = df[df['DESCRIPTION'].str.contains("Bought") |
        df['DESCRIPTION'].str.contains("Sold")]

df_out_list = []

# get a list of the tickers

tickers = list(df['SYMBOL'].unique())
tickers.pop(0)

# for each ticker calculate the desired values

for ticker in tqdm(tickers):

    # do basic calculations
    # TODO Maybe make into functions/class?

    df_ticker = df.loc[df['SYMBOL'] == ticker]
    df_div_ticker = df_dividend[df_dividend['SYMBOL'] == ticker]

    df_ticker['IsSold'] = df_ticker['DESCRIPTION'].str.contains('Sold')

    df_ticker['Current Position'] = df_ticker.apply(
        lambda row: row.QUANTITY * get_neg(row.IsSold), axis=1).values.sum()
    df_ticker['Net Dollar Position'] = df_ticker['AMOUNT'].sum()

    df_ticker['Net Price Per Share'] = df_ticker['Net Dollar Position'] / \
        df_ticker['Current Position']

    df_ticker['Total Dividend'] = df_div_ticker['AMOUNT'].sum()

    df_ticker['Average Purchase Price'] = df_ticker['PRICE'].mean()

    if grab_current_price:

        current_price = get_price(ticker, tiingo_api, df_ticker)
        # current_price = get_price_test(ticker, df)
        # TO TEST PRICE current_price = get_price_test(ticker, df)
        # do calculations
        df_ticker['Current Price'] = current_price

        df_ticker['Curr Val'] = df_ticker['Current Price'] * \
            df_ticker['Current Position']

        df_ticker['Gain/Loss'] = df_ticker['Net Dollar Position'] + \
            df_ticker['Curr Val']

        df_sub = df_ticker[['SYMBOL', 'Current Position',
                            'Net Dollar Position', 'Net Price Per Share',
                            'Current Price', 'Curr Val', 'Gain/Loss',
                            'Total Dividend', 'Average Purchase Price']]

        # only need the first result

        df_sub = df_sub.iloc[1:2, :]

        # add df to list of all ticker dfs

        df_out_list.append(df_sub)

    else:
        # only grab revelent columns

        df_sub = df_ticker[['SYMBOL', 'Current Position',
                            'Net Dollar Position', 'Net Price Per Share',
                            'Total Dividend', 'Average Purchase Price']]
        # only the first entry

        df_sub = df_sub.iloc[1:2, :]

        # add df to list of all ticker dfs

        df_out_list.append(df_sub)

# Combine into one df

df_big = pd.concat(df_out_list, axis=0)

# Clean up
print("Cleaning up...")

df_big.fillna(0, inplace=True)
df_big.replace({np.inf: 0, 'NaN': 0, 'inf': 0, '-inf': 0}, inplace=True)
df_big.sort_values(by=['SYMBOL', 'Current Position', 'Net Dollar Position'],
                   inplace=True)
df_big = df_big.round(3)

# Create filename to output file to

str_to_hash = df_big.to_string()

hashed = abs(hash(str_to_hash)) % (10 ** 8)

cwd = os.getcwd()

outpath = str(cwd) + '/' + str(out_filename) + '_' + str(hashed) \
    + '_' + str(today) + '.csv'

# save file
if output_file:

    print(f'Saving File in {outpath}...')

    df_big.to_csv(outpath, index=False)

graph_hists(df_big)
