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
    return Series, np, pd, plt, sns, xgb


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
def _(store):
    store.head()
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

    # NaN値を埋める(Open が欠損なら、平日相当=1、日曜=0 を入れる)
    # 右辺 (test["DayOfWeek"] != 7).astype(int) は、日曜=7なら0、それ以外は1。
    test.loc[test['Open'].isnull(), 'Open'] = (test['DayOfWeek'] != 7).astype(int)

    sns.countplot(x='Open', hue='DayOfWeek', data=train, palette='husl', ax=_axis1)
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

    # 最大/最小顧客数、第2、3四分位数をプロットする
    average_customers = train.groupby('Date')['Customers'].mean()
    ax3 = average_customers.plot(legend=True, marker='o', ax=_axis2)
    # 日付でグループ化し、平均顧客数を取得
    ax3.set_xticks(range(len(average_customers)))
    # 時間の経過に伴う平均顧客数をプロット
    xlabels = ax3.set_xticklabels(average_customers.index.to_list(), rotation=90)

    sns.boxplot([train['Customers']], whis=np.inf, ax=_axis1, orient='h')
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
def _(sns, train):
    # StateHoliday

    # StateHoliday列には数値0と文字列0が存在するため、値0を「0」に合わせる必要がある
    train.loc[train['StateHoliday'] == 0, 'StateHoliday'] = '0'

    sns.countplot(x='StateHoliday', data=train)
    return


@app.cell
def _(plt, sns, train):
    _fig, (_axis1, _axis2) = plt.subplots(1, 2, figsize=(16, 4))

    mask = (train['StateHoliday'] != '0') & (train['Sales'] > 0)
    sns.barplot(x='StateHoliday', y='Sales', data=train, ax=_axis1)
    sns.barplot(x='StateHoliday', y='Sales', data=train[mask], ax=_axis2)
    return


@app.cell
def _(plt, sns, test, train):
    # 平日は0、イースター/クリスマス/通常の祝日は1にマッピング
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SchoolHoliday
    """)
    return


@app.cell
def _(plt, sns, train):
    # SchoolHoliday

    sns.countplot(x="SchoolHoliday", data=train)

    _fig,(_axis1,_axis2)=plt.subplots(1,2,figsize=(16,4))

    sns.barplot(x="SchoolHoliday",y="Sales",data=train,ax=_axis1)
    sns.barplot(x="SchoolHoliday",y="Customers",data=train,ax=_axis2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Sales
    """)
    return


@app.cell
def _(np, plt, sns, train):
    # Sales

    _fig,(_axis1,_axis2)=plt.subplots(2,1,figsize=(16,8))

    # 最大/最小売り上げ、第2、3四分位数をプロットする
    sns.boxplot([train["Sales"]],whis=np.inf,ax=_axis1,orient="h")

    # 0の値がほとんど見られるのは、店舗が閉店していたため
    train["Sales"].plot(kind="hist",bins=70,xlim=(0,15000),ax=_axis2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ストアデータのデータフレーム
    """)
    return


@app.cell
def _(pd, store, train):
    # Store.csv

    average_sales_customers = train.groupby("Store",as_index=False)[["Sales","Customers"]].mean()
    sales_customers_df = average_sales_customers[["Store", "Sales", "Customers"]]

    store_df = pd.merge(sales_customers_df, store, on='Store')

    store_df.head()
    return (store_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### StoreType
    """)
    return


@app.cell
def _(plt, sns, store_df):
    # StoreType

    _fig,(_axis1,_axis2,_axis3) = plt.subplots(1,3,figsize=(24,4))

    # StoreType、および StoreType 対 平均売上高と顧客数をプロット 
    sns.countplot(x="StoreType",data=store_df,order=["a","b","c","d"],ax=_axis1)

    sns.barplot(x="StoreType",y="Sales",data=store_df,order=["a","b","c","d"],ax=_axis2)
    sns.barplot(x="StoreType",y="Customers",data=store_df,order=["a","b","c","d"],ax=_axis3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Assortment
    """)
    return


@app.cell
def _(plt, sns, store_df):
    # Assortment

    # 商品構成、および商品構成対平均売上と顧客数をプロット
    _fig,(_axis1,_axis2,_axis3)=plt.subplots(1,3,figsize=(24,4))

    sns.countplot(x="Assortment",data=store_df,order=["a","b","c"],ax=_axis1)
    sns.barplot(x="Assortment",y="Sales",data=store_df,order=["a","b","c"],ax=_axis2)
    sns.barplot(x="Assortment",y="Customers",data=store_df,order=["a","b","c"],ax=_axis3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Promo2
    """)
    return


@app.cell
def _(plt, sns, store_df):
    # Promo2

    # Promo2、およびPromo2対平均売上と顧客数をプロット
    _fig,(_axis1,_axis2,_axis3)=plt.subplots(1,3,figsize=(24,4))

    sns.countplot(x="Promo2",data=store_df,ax=_axis1)
    sns.barplot(x="Promo2",y="Sales",data=store_df,ax=_axis2)
    sns.barplot(x="Promo2",y="Customers",data=store_df,ax=_axis3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CompetitionDistance
    """)
    return


@app.cell
def _(store_df):
    # CompetitionDistance

    # null値を中央値で埋める
    store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median())

    # 競合店舗との距離対売り上げをプロットする
    store_df.plot(kind="scatter",x="CompetitionDistance",y="Sales",figsize=(16,4))
    return


@app.cell
def _(store_df):
    store_df.plot(kind="kde",x="CompetitionDistance",y="Sales",figsize=(16,4))
    return


@app.cell
def _(plt, store_df, train):
    # 競合店舗の始業が始まってから、店舗の平均売上は時間とともにどう変化したか？
    # 例：store_id = 6 の店舗では、競合開始以降、平均売上が劇的に減少している

    store_id=6
    store_data= train[train["Store"]==store_id]
    average_store_sales=store_data.groupby("Date")["Sales"].mean()

    # 競合店舗の始業年月を取得する
    year=store_df["CompetitionOpenSinceYear"].loc[store_df["Store"]==store_id].values[0]
    month=store_df["CompetitionOpenSinceMonth"].loc[store_df["Store"]==store_id].values[0]

    # プロット
    ax4=average_store_sales.plot(legend=True,figsize=(16,4),marker="o")
    ax4.set_xticks(range(len(average_store_sales)))
    ax4.set_xticklabels(average_store_sales.index.to_list(),rotation=90)

    # trainに格納されている店舗売上データは全て2013年から2015年までであるため、
    # 年が2013以上であり、かつ月と日がNaN値でないことを確認する必要がある。
    if year >= 2013 and year==year and month==month:
        ax4.axvline(x=((year-2013) * 12) + (month - 1), linewidth=3, color='grey')
    plt.show()
    return store_data, store_id


@app.cell
def _(mo):
    mo.md(r"""
    ### リスク分析
    """)
    return


@app.cell
def _(np, plt, store_data, store_df, store_id, train):
    # リスク分析
    # 店舗のリスクの分析する。リスク(標準偏差)　期待値(平均値)

    store_average=store_data["Sales"].mean()
    store_std=store_data["Sales"].std()

    plt.scatter(store_average,store_std,alpha=0.5,s=np.pi*20)

    # 店舗売上の最小値・最大値、平均値、標準偏差を取得
    # store_df["Sales"]には店舗ごとの平均売上が格納されている
    std_scales=train.groupby("Store")["Sales"].std()

    min_average=store_df["Sales"].min()
    max_average=store_df["Sales"].max()
    min_std=std_scales.min()
    max_std=std_scales.max()

    # xとyの制限を設定する
    plt.ylim([min_std,max_std])
    plt.xlim([min_average,max_average])

    # タイトル設定
    plt.xlabel("Expected Sales")
    plt.ylabel("Risk")

    # ラベルをセット
    label,x,y="Store {}".format(store_id),store_average,store_std
    plt.annotate(label,
                 xy=(x,y),xytext=(50,50),
                 textcoords="offset points",ha="right",va="bottom",
                 arrowprops=dict(arrowstyle="-",connectionstyle="arc3,rad=-0.3"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 相関の可視化
    """)
    return


@app.cell
def _(pd, train):
    # 店舗間の相関を可視化する

    store_piv = pd.pivot_table(train,values="Sales",index="Date",columns=["Store"],aggfunc="sum")

    store_pct_change = store_piv.pct_change(fill_method=None).dropna()
    store_piv.head()
    return (store_piv,)


@app.cell
def _(plt, sns, store_piv):
    # 店舗間の相関をプロット

    start_store = 1
    end_store = 5

    _fig,(_axis1) = plt.subplots(1,1,figsize=(15,5))

    # 各店舗の総売り上げ
    sns.heatmap(store_piv[list(range(start_store,end_store+1))].corr(),annot=True,linewidths=2)
    return


@app.cell
def _(pd, test, train):
    # テストデータでは2015年の8,9月しかないことに注意

    # 年月をドロップする
    train.drop(["Year","Month"],axis=1,inplace=True)
    test.drop(["Year","Month"],axis=1,inplace=True)

    # DayOfWeekのダミー変数を作る
    day_dummies_train = pd.get_dummies(train["DayOfWeek"],prefix="Day")
    day_dummies_train = day_dummies_train.drop(["Day_7"],axis=1)

    day_dummies_test = pd.get_dummies(test["DayOfWeek"],prefix="Day")
    day_dummies_test = day_dummies_test.drop(["Day_7"],axis=1)

    train_formed = train.join(day_dummies_train)
    test_formed = test.join(day_dummies_test)

    train_formed.drop(["DayOfWeek"],axis=1,inplace=True)
    test_formed.drop(["DayOfWeek"],axis=1,inplace=True)

    # 開店していない店舗は予測に寄与しないため排除する
    train_formed = train_formed[train_formed["Open"] != 0]

    # 必要のない列もドロップする。
    train_formed.drop(["Open","Customers","Date"],axis=1,inplace=True)

    # 閉店した店舗のIDを保存する。後ほどそれらの売上高を0に設定するため（下記参照）。
    closed_store_ids = test["Id"][test["Open"] == 0].values

    # 開店していない店舗のデータを排除する
    test_formed = test_formed[test_formed["Open"] != 0]

    # 必要のない列もドロップする。
    test_formed.drop(["Open","Date"],axis=1,inplace=True)
    return closed_store_ids, test_formed, train_formed


@app.cell
def _(train_formed):
    train_formed.head()
    return


@app.cell
def _(Series, closed_store_ids, pd, test_formed, train_formed, xgb):
    # 各店舗をループで処理し、現在の店舗のデータを用いてモデルを学習させ、その店舗の売上値を予測する。

    train_dict = dict(list(train_formed.groupby("Store")))
    test_dict = dict(list(test_formed.groupby("Store")))
    submission = Series()
    scores = []

    for i in test_dict:
        # 現在の店舗
        current_store = train_dict[i]

        # 訓練セットとテストセットを定義する
        X_train = current_store.drop(["Sales","Store"],axis=1)
        Y_train = current_store["Sales"]
        X_test = test_dict[i].copy()

        store_ids = X_test["Id"]
        X_test = X_test.drop(["Id","Store"],axis=1)

        # 線形回帰
        # lreg = LinearRegression()
        # lreg.fit(X_train,Y_train)
        # Y_pred = lreg.predict(X_test)
        # scores.append(lreg.score(X_train,Y_train))

        # Xgboost
        params = {"objective": "reg:linear",  "max_depth": 10}
        T_train_xgb = xgb.DMatrix(X_train, Y_train)
        X_test_xgb  = xgb.DMatrix(X_test)
        gbm = xgb.train(params, T_train_xgb, 100)
        Y_pred = gbm.predict(X_test_xgb)
    
        submission = pd.concat([submission,pd.Series(Y_pred,index=store_ids)],axis=0)
    
    submission = pd.concat([submission,pd.Series(0,index=closed_store_ids)],axis=0)

        # csvファイルに保存する
    submission = pd.DataFrame({"Id":submission.index,"Sales":submission.values})
    submission.to_csv("rossmann2.csv",index=False)
    return


if __name__ == "__main__":
    app.run()
