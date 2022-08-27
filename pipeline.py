import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import requests
import yfinance as yf
import statsmodels.formula.api as smf
from multiprocessing import Pool
import searches
import argparse
import itertools

LOOKBACK_DAYS = 30


def config_generator(param_names, search_lsts):
    for comb in itertools.product(*search_lsts):

        config = {param_names[i]: comb[i] for i in range(len(param_names))}

        yield config


def get_data_for_coin(coin):

    def get_rolling_data(data_df, config):

        dates = np.unique(data_df.index.date)

        insampler2 = []
        rollingr2 = []
        PnL = []
        PPN = []

        for i in range(15, len(dates)):

            train_dates = dates[i - 15:i]
            val_dates = dates[i]

            train_df = data_df[np.isin(data_df.index.date, train_dates)].copy()
            val_df = data_df[np.isin(data_df.index.date, val_dates)].copy()

            col_means = train_df.iloc[:, :-1].mean()
            col_stds = train_df.iloc[:, :-1].std()

            # normalization, necesssary for lasso
            train_df.iloc[:, :-1] = (train_df.iloc[:,
                                     :-1].copy() - col_means) / col_stds
            val_df.iloc[:, :-1] = (val_df.iloc[:, :-
                                   1].copy() - col_means) / col_stds

            model = smf.ols(formula="futurePxChange ~ sumOpenInterestChange + logBuySellRatio + logTopTraderAccountsLongShortRatio + logTopTraderPositionsLongShortRatio + logGlobalAccountsLongShortRatio + pastPxChange - 1", data=train_df)
            model = model.fit_regularized(
                alpha=config["reg"], L1_wt=1, refit=True)
            y_pred = model.predict(val_df)

            PnL.append(val_df[y_pred > 0]["futurePxChange"].sum() * 10000)

            notional = len(val_df[y_pred > 0]) * 10000 + (1 +
                                                          val_df[y_pred > 0]["futurePxChange"]).sum() * 10000

            PPN.append(val_df[y_pred > 0]
                       ["futurePxChange"].sum() * 10000 / notional)

            insampler2.append(model.rsquared)
            rollingr2.append(1 - np.power(val_df["futurePxChange"] - y_pred, 2).sum() / (
                np.power(val_df["futurePxChange"], 2).sum()))

        return pd.DataFrame(data=np.array([insampler2, rollingr2, PnL, PPN]).T, index=dates[15:], columns=["insample", "outOfSample", "PnL", "PPN"])

    # get dataframe

    # open interest
    oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
    # long/short ratio (account)
    ra_url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"
    # long/short ratio (positions)
    rp_url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
    # long/short ratio (global)
    rg_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    # taker buy/sell volume ratio
    tv_url = "https://fapi.binance.com/futures/data/takerlongshortRatio"

    end_datetime = pd.to_datetime(datetime.today().date())

    oi_dfs = []
    ra_dfs = []
    rp_dfs = []
    rg_dfs = []
    tv_dfs = []
    px_dfs = []

    # Binance futures only provide data for the last 30 days
    start_datetime = end_datetime - timedelta(days=LOOKBACK_DAYS)

    while start_datetime < end_datetime:

        oi_params = {
            "symbol": f"{coin}USDT",
            "period": "5m",
            "limit": 500,
            "startTime": int(start_datetime.timestamp() * 1000),
            "endTime": int((start_datetime + timedelta(days=1)).timestamp() * 1000 - 1)
        }

        ra_params = {
            "symbol": f"{coin}USDT",
            "period": "5m",
            "limit": 500,
            "startTime": int(start_datetime.timestamp() * 1000),
            "endTime": int((start_datetime + timedelta(days=1)).timestamp() * 1000 - 1)
        }

        rp_params = {
            "symbol": f"{coin}USDT",
            "period": "5m",
            "limit": 500,
            "startTime": int(start_datetime.timestamp() * 1000),
            "endTime": int((start_datetime + timedelta(days=1)).timestamp() * 1000 - 1)
        }

        rg_params = {
            "symbol": f"{coin}USDT",
            "period": "5m",
            "limit": 500,
            "startTime": int(start_datetime.timestamp() * 1000),
            "endTime": int((start_datetime + timedelta(days=1)).timestamp() * 1000 - 1)
        }

        tv_params = {
            "symbol": f"{coin}USDT",
            "period": "5m",
            "limit": 500,
            "startTime": int(start_datetime.timestamp() * 1000),
            "endTime": int((start_datetime + timedelta(days=1)).timestamp() * 1000) - 1
        }

        oi_response = requests.get(oi_url, params=oi_params)
        ra_response = requests.get(ra_url, params=ra_params)
        rp_response = requests.get(rp_url, params=rp_params)
        rg_response = requests.get(rg_url, params=rg_params)
        tv_response = requests.get(tv_url, params=tv_params)

        curr_oi_df = pd.DataFrame(oi_response.json())
        curr_ra_df = pd.DataFrame(ra_response.json())
        curr_rp_df = pd.DataFrame(rp_response.json())
        curr_rg_df = pd.DataFrame(rg_response.json())
        curr_tv_df = pd.DataFrame(tv_response.json())
        curr_px_df = yf.download(f"{coin}-USD", interval="5m", start=start_datetime,
                                 end=(start_datetime + timedelta(days=1)))

        oi_dfs.append(curr_oi_df)
        ra_dfs.append(curr_ra_df)
        rp_dfs.append(curr_rp_df)
        rg_dfs.append(curr_rg_df)
        tv_dfs.append(curr_tv_df)
        px_dfs.append(curr_px_df)

        start_datetime = start_datetime + timedelta(days=1)

    oi_df = pd.concat(oi_dfs).set_index("timestamp")
    ra_df = pd.concat(ra_dfs).set_index("timestamp")
    rp_df = pd.concat(rp_dfs).set_index("timestamp")
    rg_df = pd.concat(rg_dfs).set_index("timestamp")
    tv_df = pd.concat(tv_dfs).set_index("timestamp")
    px_df = pd.concat(px_dfs)
    px_df.index = px_df.index.astype(np.int64) // 10**6

    oi_df = oi_df[["sumOpenInterest"]]
    ra_df = ra_df[["longShortRatio"]].rename(
        columns={"longShortRatio": "topTraderAccountsLongShortRatio"})
    rp_df = rp_df[["longShortRatio"]].rename(
        columns={"longShortRatio": "topTraderPositionsLongShortRatio"})
    rg_df = rg_df[["longShortRatio"]].rename(
        columns={"longShortRatio": "globalAccountsLongShortRatio"})
    px_df = px_df[["Open"]].rename(
        columns={"Open": "price"})

    df = pd.concat([oi_df, ra_df, rp_df, rg_df, tv_df, px_df],
                   axis=1).sort_index()
    df.index = pd.to_datetime(df.index, unit="ms")
    df = df.reindex(pd.date_range(start=end_datetime - timedelta(days=LOOKBACK_DAYS),
                    end=end_datetime, freq="5min")).astype(float)

    df["sumOpenInterestChange"] = df["sumOpenInterest"].pct_change()
    df["logBuySellRatio"] = np.log(df["buySellRatio"])
    df["logTopTraderAccountsLongShortRatio"] = np.log(
        df["topTraderAccountsLongShortRatio"]
    )
    df["logTopTraderPositionsLongShortRatio"] = np.log(
        df["topTraderPositionsLongShortRatio"]
    )
    df["logGlobalAccountsLongShortRatio"] = np.log(
        df["globalAccountsLongShortRatio"]
    )
    df["pastPxChange"] = df["price"].pct_change().shift(-1)
    df["futurePxChange"] = df["price"].pct_change().shift(-2)

    df = df.drop(columns=[
        "sumOpenInterest",
        "price",
        "buySellRatio",
        "topTraderAccountsLongShortRatio",
        "topTraderPositionsLongShortRatio",
        "globalAccountsLongShortRatio",
        "buyVol",
        "sellVol",
    ])
    data_df = df.dropna()

    # hyperparameter search
    parser = argparse.ArgumentParser()
    parser.add_argument("search_name", type=str)
    args = parser.parse_args()
    search = searches.available_searches[args.search_name]()

    search_lsts = []
    param_names = []

    for param_name, vals in search.items():
        search_lsts.append(vals)
        param_names.append(param_name)

    result = {tuple(config.values()): get_rolling_data(data_df, config)
              for config in config_generator(param_names, search_lsts)}

    return pd.concat(result.values(), keys=result.keys())


if __name__ == '__main__':

    coins = searches.symbols

    with Pool() as pool:
        results = pool.imap_unordered(get_data_for_coin, coins)
        result_df = pd.concat(results, keys=coins)
        result_df.to_csv("result_by_coin.csv")
