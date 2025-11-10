import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### インポート
    """)
    return


@app.cell
def _():
    # pandas
    import pandas as pd
    from pandas import Series,DataFrame

    # numpy, matplotlib, seaborn
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    # '%matplotlib inline' command supported automatically in marimo

    # machine learning
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    import xgboost as xgb
    return np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### データ読み込み
    """)
    return


@app.cell
def _(pd):
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    store = pd.read_csv("./data/store.csv")
    return store, test, train


@app.cell
def _(train):
    train.head()
    return


@app.cell
def _(store, test, train):
    train.info()
    print("----------------------")
    test.info()
    print("----------------------")
    store.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Open
    """)
    return


@app.cell
def _(plt, sns, test, train):
    # Open
    _fig, _axis1 = plt.subplots(figsize=(16, 4))
    sns.countplot(x='Open', hue='DayOfWeek', data=train, palette='husl', ax=_axis1)
    # NaN値を埋める(Open が欠損なら、平日相当=1、日曜=0 を入れる)
    # 右辺 (test["DayOfWeek"] != 7).astype(int) は、日曜=7なら0、それ以外は1。
    test.loc[test['Open'].isnull(), 'Open'] = (test['DayOfWeek'] != 7).astype(int)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Date
    """)
    return


@app.cell
def _(test, train):
    # Date

    # 年と月の列を作る
    train["Year"] = train["Date"].apply(lambda x: int(str(x)[:4]))
    train["Month"] = train["Date"].apply(lambda x: int(str(x)[5:7]))

    test["Year"] = test["Date"].apply(lambda x: int(str(x)[:4]))
    test["Month"] = test["Date"].apply(lambda x: int(str(x)[5:7]))

    # 年月日を年月に入れ替える
    train["Date"] = train["Date"].apply(lambda x: (str(x)[:7]))
    test["Date"] = test["Date"].apply(lambda x: (str(x)[:7]))

    # Dateでグルーピングして、平均売上とそのパーセンテージを取得する
    average_sales = train.groupby("Date")["Sales"].mean()
    pct_change_sales = train.groupby("Date")["Sales"].sum().pct_change()
    return average_sales, pct_change_sales


@app.cell
def _(average_sales, pct_change_sales, plt):
    _fig, (_axis1, _axis2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8))
    ax1 = average_sales.plot(legend=True, ax=_axis1, marker='o', title='Average Sales')
    # 時系列の売上平均をプロットする
    ax1.set_xticks(range(len(average_sales)))
    ax1.set_xticklabels(average_sales.index.to_list(), rotation=90)
    # 時系列の売上平均前日比(%)をプロットする
    # ax2.set_xticks(range(len(pct_change_sales)))
    # ax2.set_xticklabels(pct_change_sales.index.tolist(), rotation=90)
    ax2 = pct_change_sales.plot(legend=True, ax=_axis2, marker='o', rot=90, colormap='summer', title='Sales Percent Change')
    return


@app.cell
def _(plt, sns, train):
    # 年系列で売り上げ平均と顧客数をプロットする
    _fig, (_axis1, _axis2) = plt.subplots(1, 2, figsize=(16, 4))
    sns.barplot(x='Year', y='Sales', data=train, ax=_axis1)
    sns.barplot(x='Year', y='Customers', data=train, ax=_axis2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Customers
    """)
    return


@app.cell
def _(np, plt, sns, train):
    # Customers
    _fig, (_axis1, _axis2) = plt.subplots(2, 1, figsize=(16, 8))
    sns.boxplot([train['Customers']], whis=np.inf, ax=_axis1, orient='h')
    # 最大/最小顧客数、第2、3四分位数をプロットする
    average_customers = train.groupby('Date')['Customers'].mean()
    ax = average_customers.plot(legend=True, marker='o', ax=_axis2)
    # 日付でグループ化し、平均顧客数を取得
    ax.set_xticks(range(len(average_customers)))
    # 時間の経過に伴う平均顧客数をプロット
    xlabels = ax.set_xticklabels(average_customers.index.to_list(), rotation=90)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DayOfWeek
    """)
    return


@app.cell
def _(plt, sns, train):
    # DayOfWeek
    _fig, (_axis1, _axis2) = plt.subplots(1, 2, figsize=(16, 4))
    # 店舗が閉店時および開店時双方のケース
    sns.barplot(x='DayOfWeek', y='Sales', data=train, order=[1, 2, 3, 4, 5, 6, 7], ax=_axis1)
    sns.barplot(x='DayOfWeek', y='Customers', data=train, order=[1, 2, 3, 4, 5, 6, 7], ax=_axis2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Promo
    """)
    return


@app.cell
def _(plt, sns, train):
    # Promo
    _fig, (_axis1, _axis2) = plt.subplots(1, 2, figsize=(16, 4))
    # 販促有無別の平均売上高と顧客数をプロット
    sns.barplot(x='Promo', y='Sales', data=train, ax=_axis1)
    sns.barplot(x='Promo', y='Customers', data=train, ax=_axis2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### StateHoliday
    """)
    return


@app.cell
def _(plt, sns, train):
    # StateHoliday
    train.loc[train['StateHoliday'] == 0, 'StateHoliday'] = '0'
    # StateHoliday列には数値0と文字列0が存在するため、値0を「0」に合わせる必要がある
    sns.countplot(x='StateHoliday', data=train)
    _fig, (_axis1, _axis2) = plt.subplots(1, 2, figsize=(16, 4))
    sns.barplot(x='StateHoliday', y='Sales', data=train, ax=_axis1)
    mask = (train['StateHoliday'] != '0') & (train['Sales'] > 0)
    sns.barplot(x='StateHoliday', y='Sales', data=train[mask], ax=_axis2)
    return


@app.cell
def _(plt, sns, test, train):
    # 平日は0、イースター/クリスマス/通常の祝日は1にマッピング。
    train['StateHoliday'] = train['StateHoliday'].map({0: 0, '0': 0, 'a': 1, 'b': 1, 'c': 1})
    test['StateHoliday'] = test['StateHoliday'].map({0: 0, '0': 0, 'a': 1, 'b': 1, 'c': 1})
    _fig, (_axis1, _axis2) = plt.subplots(1, 2, figsize=(16, 4))
    sns.barplot(x='StateHoliday', y='Sales', data=train, ax=_axis1)
    sns.barplot(x='StateHoliday', y='Customers', data=train, ax=_axis2)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
