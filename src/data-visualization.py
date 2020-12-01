from os import listdir
from os.path import isfile, join
import pandas as pd
import math
import datetime
import seaborn as sns
from matplotlib import pyplot as plt

fill_color = ["#d7ffad"]
edge_color = '#000066'

df = pd.read_csv('./work-data/train-data-cleaned.csv')


def plot_revenue_by_month(data, filename):
    byMonth = data.groupby(['month', 'year']).sum()
    byMonth = byMonth.reset_index()
    byMonth = byMonth.sort_values(by=['year', 'month'], axis=0)
    byMonth.rename(columns={'price': 'revenue'}, inplace=True)
    byMonth = byMonth[['month', 'year', 'revenue']]
    byMonth['year_month'] = byMonth.year*100+byMonth.month

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    revenue_by_month = sns.barplot(
        x=byMonth.year_month, y=byMonth.revenue, palette=sns.color_palette(fill_color), edgecolor=edge_color)
    for item in revenue_by_month.get_xticklabels():
        item.set_rotation(45)
    revenue_by_month.set_title('revenue by month')

    plt.savefig(filename)


def plot_views_by_country(data, filename):
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    dfv = df[['country', 'price']]
    dfv = dfv.groupby(['country']).count()
    dfv = dfv.reset_index()
    dfv.rename(columns={'price': 'views'}, inplace=True)
    dfv = dfv.sort_values(by=['views'], ascending=False)
    dfv = dfv[dfv.views > 1500]
    views_by_country = sns.barplot(
        data=dfv, x='country', y='views', palette=sns.color_palette(fill_color), edgecolor=edge_color)
    for item in views_by_country.get_xticklabels():
        item.set_rotation(45)
    views_by_country.set_title('views by country')
    plt.savefig(filename)


plot_revenue_by_month(df, './imgs/revenue_by_month.png')
plot_views_by_country(df, './imgs/views_by_country.png')
