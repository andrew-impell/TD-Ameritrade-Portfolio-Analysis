import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('fivethirtyeight')


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


df = pd.read_csv('portfolio_stats_27264959_2020-04-11.csv', header=0)

graph_hists(df)
