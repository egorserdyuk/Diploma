import copy
import datetime
import math
import os
import urllib
import zipfile

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from numpy import nan as Nan
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale, normalize
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import ReducedForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import relative_loss
from sktime.utils.plotting import plot_series

# todo states where cases and deaths are most and least correlated


st.set_page_config(
    page_title="Корреляционный анализ показателей регионов алтайского края",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache(suppress_st_warning=True)
def process_data(all_states, state):
    """
    Process CSVs. Smooth and compute new series.

    :param all_states: Boolean if "all states" is checked
    :param state: Selected US state
    :return: Dataframe
    """
    # Data
    if all_states:
        df = pd.read_csv("outputAll.csv").sort_values("год", ascending=True).reset_index()
    else:
        df = (
            pd.read_csv("outputForWeb.csv")
            .sort_values("год", ascending=True)
            .reset_index()
            .query('регион=="{}"'.format(state))
        )
    #df = df.query("date >= 20200301")
    #df["год"] = df["год"].astype('int')
    df.set_index("год", inplace=True)
    df = rename_columns(df)

    if np.inf in df.values:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


@st.cache()
def find_max_correlation(col, col2):
    """
    Take two series and test all alignments for maximum correlation.
    :param col: Column 1
    :param col2: Column 2
    :return: Best r, best shift
    """
    correl = col.corr(col2)
    best_cor = correl
    best_i = 1

    return best_cor, best_i

def plot_cor(col, col2, best_i, best_cor):
    """
    Plot interactive chart showing correlation between two shifted series.

    :param col:
    :param col2:
    :param best_i:
    :param best_cor:
    """
    # st.line_chart({col.name: col.shift(best_i), col2.name: col2})
    st.text(
        "{} и {}\nзначение корреляции r = {}".format(
            col.name, col2.name, round(best_cor, 2)
        )
    )

    # altair chart
    src = pd.DataFrame({col.name: col, col2.name: col2}).reset_index()
    base = alt.Chart(src).encode(alt.X("год:T", axis=alt.Axis(title=None)))

    line = base.mark_line(stroke="orange").encode(
        alt.Y(col.name, axis=alt.Axis(title=col.name, titleColor="orange"))
    )

    line2 = base.mark_line(stroke="#5276A7").encode(
        alt.Y(col2.name, axis=alt.Axis(title=col2.name, titleColor="#5276A7"))
    )

    chrt = alt.layer(line, line2).resolve_scale(y="independent")
    st.altair_chart(chrt, use_container_width=True)


# @st.cache(ttl=TTL)
def get_shifted_correlations(df, cols):
    """
    Interactive correlation explorer. For two series, finds the alignment that maximizes correlation.
    :param df:
    :param cols:
    :return:
    """
    a = st.selectbox("Does this", cols, index=3)
    b = st.selectbox("Correlate with this?", cols, index=2)
    lb = st.slider(
        "Выберите временной промежуток для корреляционного анализа",
        min_value=int(df.index.min()) + 1,
        max_value=int(df.index.max()),
        value=int(df.index.max()),
        step=1,
        format="%d год",
        key="window2",
    )
    test = -(lb - int(df.index.min())) - 1
    data = df[b].iloc[-(lb - int(df.index.min())) - 1:]
    cor, shift = find_max_correlation(df[a].iloc[-lb:], df[b].iloc[-(lb - int(df.index.min())) - 1:])
    col1, col2 = df[a].iloc[-lb:], df[b].iloc[-lb:]
    plot_cor(df[a].iloc[-lb:], df[b].iloc[-lb:], shift, cor)

    return cols, a, b, lb


@st.cache()
def get_cor_table(cols, lb, df):
    """
    Generates dataframe of correlated series and alignments for all given columns.
    :param cols:
    :param lb: Lookback for correlation coefficent
    :param df:
    :return:
    """
    # Find max
    shifted_cors = pd.DataFrame(columns=["a", "b", "r", "shift"])
    for i in cols:
        for j in cols:
            if i == j:
                continue
            cor, shift_temp = find_max_correlation(df[i].iloc[-lb:], df[j].iloc[-lb:])
            shifted_cors = shifted_cors.append(
                {"a": i, "b": j, "r": cor, "shift": shift_temp}, ignore_index=True
            )
    return shifted_cors


def forecast_ui(cors_df, lookback):
    """
    Gets user input for correlation forecast
    :param cors_df: Correlations table
    :return:
    """
    # st.header('Forecast Based on Shifted Correlations')

    cors_df = cors_df.query("r >0.5 and shift >0")
    if len(cors_df) < 2:
        cors_df = cors_df.query("r >0.0 and shift >=0")
        st.warning(
            "Few strong correlations found for forecasting. Try adjusting lookback window."
        )

    # forecast_len = int(np.mean(cors_df['r'].values * cors_df['shift'].values))
    # st.write("Forecast Length = average shift weighted by average correlation = ", forecast_len)
    days_back = (
        -st.slider(
            "See how past forecasts did:", 0, lookback // 2, 0, format="%d days back"
        )
        - 1
    )
    return days_back


@st.cache()
def compute_weighted_forecast(days_back, b, shifted_cors):
    """
    Computes a weighted average of all series that correlate with column B when shifted into the future.
    The weighted average is scaled and aligned to the target column b.

    :param days_back: How far back to start forecasting.
    :param b: Target column to forecast.
    :param shifted_cors: Table of correlated series and shifts
    :return:
    """
    cors_df = shifted_cors.query("b == '{}' and r >0.5 and shift >0".format(b))
    if len(cors_df) < 3:
        cors_df = shifted_cors.query("b == '{}' and r >0.0 and shift >0".format(b))
        # st.warning("No strong correlations found for forecasting. Try adjusting lookback window.")
        # st.stop()
    cols = cors_df["a"].values
    dfcols = df[cols]
    dfmin = df[b].min()
    dfmax = df[b].max()
    # scale to predicted val
    df[cols] = minmax_scale(df[cols], (df[b].min(), df[b].max()))
    for i, row in cors_df.iterrows():
        col = row["a"]
        # weight by cor
        df[col] = df[col] * row["r"]
        # shift on x axis
        df[col] = df[col].shift(row["shift"])
        # OLS
        model = sm.OLS(
            df[b].interpolate(limit_direction="both"),
            df[col].interpolate(limit_direction="both"),
        )  # Y,X or X,Y ?
        results = model.fit()
        df[col] = df[col] * results.params[0]

    forecast_len = int(np.mean(cors_df["r"].values * cors_df["shift"].values))
    forecast = df[cols].mean(axis=1)

    # ML forecast
    # forecast = ml_regression(df[cols], df[b],7)
    # df['forecast'] = forecast
    # forecast = df['forecast']

    # OLS
    df["forecast"] = forecast
    model = sm.OLS(
        df[b].interpolate(limit_direction="both"),
        df["forecast"].interpolate(limit_direction="both"),
    )  # Y,X or X,Y ?
    results = model.fit()
    # st.write('OLS Beta =', results.params)
    forecast = forecast * results.params[0]

    # Align on Y axis
    dif = df[b].iloc[days_back] - forecast.iloc[-forecast_len + days_back]
    forecast += dif

    # only plot forward forecast
    forecast.iloc[: -forecast_len + days_back] = np.NAN
    forecast.iloc[days_back:] = np.NAN

    df["forecast"] = forecast

    lines = {
        b: df[b].append(
            pd.Series([Nan for i in range(forecast_len)]), ignore_index=True
        ),
        "Forecast": df["forecast"]
        .append(pd.Series([Nan for i in range(forecast_len)]), ignore_index=True)
        .shift(forecast_len),
    }
    return lines, cors_df


def plot_forecast(lines, cors_table):
    """
    Plots output from compute_weighted_forecast()
    :param lines: Dict with forecast and target variable.
    :param cors_table: Table of correlated series and shifts
    """
    idx = pd.date_range(start=df.index[0], periods=len(lines[b]))
    df2 = pd.DataFrame(lines).set_index(idx)
    st.line_chart(df2, use_container_width=True)
    # plt.style.use('bmh')
    # st.write(df2.plot().get_figure())


@st.cache()
def compute_arima(df, colname, days, oos):
    """
    Must do computation in separate function for streamlit caching.

    :param df:
    :param colname:
    :param days:
    :param oos: Out of sample forecast.
    :return:
    """
    y = df[colname].dropna()
    if oos:
        # Forecast OOS
        range = pd.date_range(
            start=y.index[-1] + datetime.timedelta(days=1),
            end=y.index[-1] + datetime.timedelta(days=days),
        )
        fh = ForecastingHorizon(range, is_relative=False)
        forecaster = AutoARIMA(suppress_warnings=True)
        forecaster.fit(y)
        alpha = 0.05  # 95% prediction intervals
        y_pred, pred_ints = forecaster.predict(fh, return_pred_int=True, alpha=alpha)
        return [y, y_pred], ["y", "y_pred"], pred_ints, alpha
    else:
        y_train, y_test = temporal_train_test_split(y, test_size=days)
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        forecaster = AutoARIMA(suppress_warnings=True)
        forecaster.fit(y_train)
        alpha = 0.05  # 95% prediction intervals
        y_pred, pred_ints = forecaster.predict(fh, return_pred_int=True, alpha=alpha)
        return (
            [y_train, y_test, y_pred],
            ["y_train", "y_test", "y_pred"],
            pred_ints,
            alpha,
        )


def timeseries_forecast(df, colname, days=14):
    """
    ARIMA forecast wrapper

    :param df: Dataframe from process_data()
    :param colname: Name of forecasted variable
    :param days_back: Lookback when validating, and lookahead for out of sample forecast.
    """
    st.subheader("Past Performance")
    sktime_plot(*compute_arima(df, colname, days, False))

    st.subheader("Forecast")
    sktime_plot(*compute_arima(df, colname, days, True))
    # y_pred, _, pred_ints, _ = compute_arima(df, colname, days, True)
    # st.line_chart(pd.DataFrame(y_pred).transpose())


def sktime_plot(series, labels, pred_ints, alpha):
    """
    Plot forecasts using sktime plot_series
    https://docs.streamlit.io/en/stable/deploy_streamlit_app.html#limitations-and-known-issues

    :param series:
    :param labels:
    :param pred_ints:
    :param alpha:
    """

    _lock = RendererAgg.lock

    with _lock:
        fig, ax = plot_series(*series, labels=labels)
        # Plot with intervals
        ax.fill_between(
            ax.get_lines()[-1].get_xdata(),
            pred_ints["lower"],
            pred_ints["upper"],
            alpha=0.2,
            color=ax.get_lines()[-1].get_c(),
            label=f"{1 - alpha}% prediction intervals",
        )
        ax.legend()
        st.pyplot(fig)


def matplotlib_charts(df, cols):
    plt.style.use("seaborn")
    # plt.style.use("seaborn-whitegrid")
    # plt.style.use("fivethirtyeight")
    # st.pyplot(df[cols].plot.area().get_figure())

    # st.pyplot(df[[
    #             "Remaining Population",
    #             "Cumulative Recovered Infections Estimate",
    #             "First Doses Administered",]
    #         ].plot.area().get_figure())

    plots = df[cols].plot.line(subplots=True)
    st.pyplot(plots[0].get_figure())

    # plots = df[cols].plot(
    #     subplots=True, layout=(2, 2)
    # )
    # st.pyplot(plots[0][0].get_figure())

def rename_columns(df):
    # todo
    col_map = {
        "женщин": "Совершивших_преступления_женщин",
        "незанятыеРаб": "Незанятые_граждане_ищущие_работу",
        "работниковСх": "Работников_сельского_хозяйства",
        "рабочих": "Совершивших_преступления_рабочих",
        "служащих": "Совершивших_преступления_служащих",
        "трудоспосНас": "Трудоспособное_население_в_трудоспособном_возрасте",
        "освобожденоОтУг": "Освобождено_от_уголовной_ответственности",
        "привлеченоУг": "Привлечено_к_уголовной_ответственности",
        "всего": "Совершивших_преступления_всего",
    }
    df = df.rename(columns=col_map)

    cols = list(df.columns)
    return df

# Unused functions below. May use in future. ---------------------------------------------------------------------------
@st.cache()
def ml_regression(X, y, lookahead=7):
    """
    Feed correlated and shifted variables into ML model for forecasting.
    Doesn't seem to do better than weighted average forecast.

    :param X: Correlation table. df[cols]
    :param y: Target series. df[b]
    :param lookahead: Forecast this many days ahead.
    :return: Forecasted series.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    y_shift = y.shift(lookahead)
    X.fillna(0, inplace=True)
    y_shift.fillna(0, inplace=True)
    # X.interpolate(inplace=True, limit_direction='both')
    # y.interpolate(inplace=True, limit_direction='both')
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shift, random_state=0, shuffle=False
    )
    reg = GradientBoostingRegressor(random_state=0, verbose=True)
    # reg = RandomForestRegressor(random_state=0, verbose=True)
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)

    score = reg.score(X_test, y_test)
    reg.fit(X, y_shift)

    # sktime
    y.fillna(0, inplace=True)
    y_train, y_test = temporal_train_test_split(y, test_size=14)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = ReducedForecaster(
        regressor=reg, window_length=12, strategy="recursive"
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    fig, ax = plot_series(
        y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]
    )
    st.write(fig)
    relative_loss(y_test, y_pred)

    return reg.predict(X)


def get_correlations(df, cols):
    st.header("Correlations")
    df = df[cols]
    cor_table = df.corr(method="pearson", min_periods=30)
    st.write(cor_table)
    max_r = 0
    max_idx = None
    seen = []
    cors = pd.DataFrame(columns=["a", "b", "r"])
    for i in cor_table.index:
        for j in cor_table.index:
            if i == j or i == "index" or j == "index":
                continue
            if cor_table.loc[i, j] == 1:
                continue
            if cor_table.loc[i, j] > max_r:
                max_idx = (i, j)
                max_r = max(cor_table.loc[i, j], max_r)
            if (j, i) not in seen:
                cors = cors.append(
                    {"a": i, "b": j, "r": cor_table.loc[i, j]}, ignore_index=True
                )
                seen.append((i, j))
    st.write(max_idx, max_r)
    st.write(cors.sort_values("r", ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    # todo global cols lists. One for cors and one for UI
    cols = [
        "Совершивших_преступления_женщин",
        "Незанятые_граждане_ищущие_работу",
        "Работников_сельского_хозяйства",
        "Совершивших_преступления_рабочих",
        "Совершивших_преступления_служащих",
        "Трудоспособное_население_в_трудоспособном_возрасте",
        "Освобождено_от_уголовной_ответственности",
        "Привлечено_к_уголовной_ответственности",
        "Совершивших_преступления_всего"
    ]

    # Disabled data download due to the end of covidtracking.com
    # https://covidtracking.com/analysis-updates/covid-tracking-project-end-march-7
    # todo switch to Johns Hopkins Github repo
    # download_data(wait_hours=4)

    w, h, = (
        900,
        400,
    )
    states = pd.read_csv("outputForWeb.csv")["регион"].unique()

    with st.sidebar:
        st.title("Регионы алатайского края")
        st.subheader("Выберите страницу ниже:")
        mode = st.radio(
            "Меню",
            [
                "Корреляционный обозреватель",
                "Прогноз корреляций",
            ],
        )
        st.subheader("Выбрать регион или все регионы:")
        all_states = st.checkbox("Все регионы", False)
        locations = np.append(["Алтайский край всего"], states)
        state = st.selectbox("Регион", states)

    # https://docs.streamlit.io/en/stable/troubleshooting/caching_issues.html#how-to-fix-the-cached-object-mutated-warning
    df = copy.deepcopy(process_data(all_states, state))
    df_arima = copy.deepcopy(df)

    if mode == "Прогноз корреляций":
        st.title("Прогноз корреляций")
        # df,cols= rename_columns(df)
        b = st.selectbox("Выберите переменную:", cols, index=2)
        # lookback = st.slider('How far back should we look for correlations?', min_value=0, max_value=len(df),
        #                      value=len(df) - 70,
        #                      step=10, format="%d days")
        lookback = len(df) - 70
        cors_df = get_cor_table(cols, lookback, df)

        days_back = forecast_ui(cors_df, lookback)
        lines, cors_table = compute_weighted_forecast(days_back, b, cors_df)

        if len(cors_table) < 3:
            st.warning(
                "Few correlations found. Forecast may not be accurate. Try another variable."
            )

        plot_forecast(lines, cors_table)

        st.markdown(
            """
        ## How is this forecast made?

        This forecast is a weighted average of variables from the table below. $shift$ is the number of days $a$ is shifted forward, and $r$ is the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between shifted $a$ and $b$.
        """
        )
        st.write(cors_table)

        st.markdown(
            """ 
        ## Further Explanation
        The model searches every combination of $a$, $b$, and $shift$ for the highest $r$ values. Only correlations $>0.5$ are used. $r$ is used to weight each component of the forecast, and each component is scaled and aligned to the forecasted variable $b$. The forecast length is the average $shift$ weighted by the average $r$.

        Ordinary Least Squares regression is also used to scale each series from the *a* column as well as the final forecast.
        """
        )
    elif mode == "Корреляционный обозреватель":
        st.title("Интерактивный корреляционный обозреватель")
        st.write("Choose two variables and see if they are correlated.")
        cols, a, b, lookback = get_shifted_correlations(df, cols)