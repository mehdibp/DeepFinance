import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import MetaTrader5 as mt5


# -------------------------------------------------------------------------------------------
class MarketFeatureBuilder():
    def __init__(self, symbol: str, timeframe: int, date_from: datetime, date_to: datetime):

        self.symbol    = symbol
        self.timeframe = timeframe
        self.date_from = date_from
        self.date_to   = date_to

        self.data: pd.DataFrame | None = None


    # ---------------------------------------------------------------------------------------
    def fetch_MT5_data(self):
        """
		Retrieve price data from MetaTrader5 over a specified time range and save it to a CSV file.

		Parameters
		----------
		symbol    : str  -  Trading symbol (e.g., 'EURUSD')
		timeframe : int  -  Data timeframe (e.g., mt5.TIMEFRAME_M5 for 5 minutes)
		date_from : datetime  -  Start date of the range
		date_to   : datetime  -  End date of the range

		Returns
		-------
		None
			Data is saved as a CSV file. If no data is found or an error occurs,
			an error message is printed.

		Notes
		-----
			- First, a connection to MT5 is established (mt5.initialize).
			- Data is retrieved using the mt5.copy_rates_range function.
			- If successful, the data is converted to a DataFrame and the 'time' column is converted to datetime.
			- The CSV file is saved with a name including the timeframe and date range.
			- Finally, the connection to MT5 is closed (mt5.shutdown).
    	"""

        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")
            mt5.shutdown()
        else:

            ranges = [(self.date_from, self.date_to)]     # List of intervals
            dfs: list[pd.DataFrame] = []

            max_splits = 5
            # As long as we have a range and the number of divisions does not exceed the allowed limit
            while ranges and max_splits > 0:    
                new_ranges = []
                for start, end in ranges:
                    rates = mt5.copy_rates_range(self.symbol, self.timeframe, start, end)

                    if rates is not None and len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        dfs.append(df)
                    else:
                        # If no data is returned, halve the interval
                        mid = start + (end - start) / 2
                        new_ranges.append((start, mid))
                        new_ranges.append((mid, end))

                ranges = new_ranges
                max_splits -= 1

            mt5.shutdown()

            if not dfs: 
                raise RuntimeError(f"Failed to fetch data or no data found. Error: {mt5.last_error()}")
            else:
                self.data = pd.concat(dfs).drop_duplicates().sort_values(by="time").reset_index(drop=True)
                return self
            
    # ---------------------------------------------------------------------------------------
    def load_csv(self, path=None, name=None):
        location = f'./' if not path else path
        filename = f'{self.symbol} -- {self.timeframe} ({self.date_from} - {self.date_to}).csv' if not name else name
        self.data = pd.read_csv(f"{location}{filename}") 
        self.data['time'] = pd.to_datetime(self.data['time'])
        return self
    
    # ---------------------------------------------------------------------------------------
    def save(self, path=None, name=None):
        location = f'./' if not path else path
        filename = f'{self.symbol} -- {self.timeframe} ({self.date_from} - {self.date_to}).csv'.replace(" 00:00:00+00:00", "") if not name else name
        self.data.to_csv(f"{location}{filename}", index=False)
        print(f"Data saved as {location}{filename}")
    
    # ---------------------------------------------------------------------------------------
    def get_data(self):
        return self.data
    
    # ---------------------------------------------------------------------------------------
    def core_features(self):
        """
        Compute basic candlestick and time series features for financial data analysis.

        Parameters
        ----------
        df: DataFrame  -  containing the columns ['time','open','high','low','close','tick_volume']

        Returns
        -------
        DataFrame
            DataFrame with new columns:
            - return          : logarithmic return (ln(Close_t / Close_{t-1}))
            - body            : candlestick body (Close - Open)
            - range           : candlestick range (High - Low)
            - upper_wick      : upper shadow (High - max(Open, Close))
            - lower_wick      : lower shadow (min(Open, Close) - Low)
            - ema_20          : 20-period exponential moving average
            - price_ema_ratio : price-to-EMA ratio (trend strength)
            - volatility_20   : 20-period volatility (standard deviation of returns)
            - volume_norm     : normalized volume relative to 20-period average
            - hour            : hour of day extracted from the time column
        """

        df = self.data.copy()

        # Time of Day in a Cycle. ( Sin(time), Cos(time) )
        time_in_minutes  = df['time'].dt.hour*60 + df['time'].dt.minute
        time_in_radians  = 2 * np.pi * time_in_minutes / 1440
        df["time_sin"]   = np.round(np.sin(time_in_radians), 3)
        df["time_cos"]   = np.round(np.cos(time_in_radians), 3)

        df['return']     = np.log(df['close'] / df['close'].shift(1))       # r_{t}​ = ln (Close_{t} − ​Close_{t-1}​​)
        df['body'  ]     = df['close'] - df['open']                         # body_{t} ​= Close_{t} ​− Open_{t}
        df['range' ]     = df['high' ] - df['low' ]                         # range_{t} ​= High_{t} ​− Low_{t}​
        df['upper_wick'] = df['high' ] - df[['open', 'close']].max(axis=1)  # upper = High − max(Open, Close)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']    # lower = min(Open, Close) − Low

        df['ema_20']     = df['close'].ewm(span=20, adjust=False).mean()    # EMA_{t} ​= α⋅Close_{t} ​+ (1−α)EMA_{t} − 1​ , α=2/(N+1)​
        df['price_ema_ratio'] = df['close'] / df['ema_20']                  # trend_strength = Close/EMA​

        df['volatility_20'  ] = df['return'].rolling(20).std()              # σ{t} ​= sqrt( Var(r_{t−n:t}​) )
        df['volume_norm'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()   	# vol_norm = V/MA(V)
        
        self.data = df
        return self

    # ---------------------------------------------------------------------------------------
    def level_features(self, period: list[pd.Series], atr_window: int=14, digits: int=2, ema_span :int=12*12):
        """
        Perform three calculations on the DataFrame:
        1. Distance of the candle shadow from the boundaries of the previous period (day or week)
        2. Distance from rounded numbers
        3. EMA and price-to-EMA ratio (trend)
        """

        df = self.data.copy()

        df['dist_round']        = self._calc_round_distance (df, digits=digits)
        df[f'trend_{ema_span}'] = self._calc_add_trend      (df, ema_span=ema_span)
        for p in period: df[f'dist_shadow_prev_{p.name}'] = self._calc_shadow_distance(df, period=p, atr_window=atr_window)

        ### 
        if 'dist_shadow_prev_None' in df.columns:
            df = df.rename(columns={"dist_shadow_prev_None": "dist_shadow_prev_week"})
            df = df.dropna(subset =['dist_shadow_prev_week']).reset_index(drop=True)
        elif 'dist_shadow_prev_time' in df.columns:
            df = df.dropna(subset =['dist_shadow_prev_time']).reset_index(drop=True)
        ###

        self.data = df
        return self

    # ---------------------------------------------------------------------------------------
    def target_label(self, horizon: int=24, regime_threshold: float=1.0, sl_mult: float=1.0, tp_mult: float=2.0):
    
        df = self.data.copy()

        df["market_regime"] = self._calc_market_regime(df, horizon+6, regime_threshold)
        df["trade_outcome"] = self._calc_trade_outcome(df, horizon, sl_mult, tp_mult)
        df["MFE_norm"], df["MAE_norm"] = self._calc_mfe_mae (df, horizon)

        self.data = df
        return self
        

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _calc_round_distance(df: pd.DataFrame, digits: int=2):
        """
        Calculate the distance of the candle's high and low from the nearest rounded number.

        Parameters:
        ----------
        df : DataFrame containing the columns ['high','low']
        digits : Number of decimal places for rounding (default: 2)

        Output:
        -------
        DataFrame with the following columns:
        - dist_high : distance from high to its nearest rounded number
        - dist_low  : distance from low to its nearest rounded number
        - dist_round: minimum distance between high and low to a rounded number
        """

        df = df.copy()

        factor = 10**digits
        dist_high = (df['high'] - df['high'].round(digits)) * factor
        dist_low  = (df['low' ] - df['low' ].round(digits)) * factor

        dist_round = np.minimum(dist_high.abs(), dist_low.abs())
        return dist_round

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _calc_add_trend(df: pd.DataFrame, ema_span: int=12*12):
        """
        Calculate EMA and the price-to-EMA ratio as a trend indicator.

        Parameters:
        ----------
        df : DataFrame containing the 'close' column
        ema_span : EMA window length (default: 12*12, approximately 1 hour for 5-minute data)
        col_close : Name of the closing price column (default: 'close')

        Output:
        -------
        DataFrame with the following columns:
        - ema_{ema_span}   : calculated EMA
        - trend_{ema_span} : ratio of close to EMA
        """

        df = df.copy()

        ema_values = df['close'].ewm(span=ema_span).mean()
        trend_col  = df['close'] / ema_values

        return trend_col

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _calc_shadow_distance(df: pd.DataFrame, period: list[pd.Series], atr_window: int=14):
        """
        Calculate the distance of the candle shadow (high/low) from the maximum and minimum of the previous period,
        normalized by ATR.

        Parameters:
        ----------
        df : DataFrame containing the columns ['time','open','high','low','close']
        period : 'day' or 'week' → determines whether the comparison reference is the previous day or the previous week
        atr_window : ATR window length (default: 14)

        Output:
        -------
        DataFrame with columns for the distances and the minimum shadow distance to the previous range
        """

        df = df.copy()

        # Average True Range Calculate
        high_low   = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close  = np.abs(df['low' ] - df['close'].shift(1))

        tr  = np.maximum(high_low, np.maximum(high_close, low_close)) 	# TR
        atr = tr.rolling(atr_window).mean()                     		# ATR

        # Maximum and minimum period
        agg = df.groupby(period).agg(ref_high=('high','max'), ref_low=('low','min'))
        df = df.join(agg.shift(1), on=period)               # Add previous period

        # Distance of today's shadow from the boundaries of the previous period
        df['high_to_prev_high'] = (df['high'] - df['ref_high']).abs() / atr
        df['high_to_prev_low' ] = (df['high'] - df['ref_low' ]).abs() / atr
        df['low_to_prev_high' ] = (df['low' ] - df['ref_high']).abs() / atr
        df['low_to_prev_low'  ] = (df['low' ] - df['ref_low' ]).abs() / atr

        # Minimum shadow distance to previous range
        dist_shadow_prev = df[['high_to_prev_high', 'high_to_prev_low', 'low_to_prev_high', 'low_to_prev_low']].min(axis=1)
        return dist_shadow_prev


    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _calc_market_regime(df: pd.DataFrame, horizon=30, threshold=1.0):
        """
        Label market regime based on future trend strength.

        Formula:
            trend_strength = log(C[t+H] / C[t]) / volatility[t]

        Parameters
        ----------
        df : DataFrame -- containing price and volatility columns
        horizon : int, default=30 -- Prediction horizon (number of candles ahead to compute future return)
        threshold : float, default=1.0 -- Trend strength threshold for identifying UpTrend or DownTrend
        price_col : str, default="close" -- Name of the closing price column
        vol_col : str, default="volatility_20" -- Name of the volatility column (standard deviation of returns)

        Returns
        -------
        DataFrame -- with a new column:
            - market_regime ∈ {-1, 0, +1}
            -1 → DownTrend
            0 → Range
            +1 → UpTrend
        """

        df = df.copy()

        future_close   = df['close'].shift(-horizon)
        future_return  = np.log(future_close / df['close'])     # log(close[t+H] / close[t])
        trend_strength = future_return / df['volatility_20']    # ​log(C_{t+H}​/C_{t}​)​ / volatility_{t}

        # regime labeling: market_regime ∈ {-1, 0, +1} ~= {DownTrend, Range, UpTrend}
        regime = pd.Series(0, index=df.index, name="market_regime")
        regime[trend_strength >  threshold] =  1
        regime[trend_strength < -threshold] = -1

        return regime

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _calc_trade_outcome(df: pd.DataFrame, horizon=20, sl_mult=1.0, tp_mult=2.0):
        """
        Label trade outcome based on trade direction, take-profit (TP), and stop-loss (SL).

        Logic:
            - If the trade is BUY:
                TP = entry + tp_mult * volatility
                SL = entry - sl_mult * volatility
            - If the trade is SELL:
                TP = entry - tp_mult * volatility
                SL = entry + sl_mult * volatility
            - Within the next horizon candles:
                If SL is hit → outcome = -1
                If TP is hit → outcome = +1
                If neither is hit → outcome = 0
            - If direction is in range → NaN

        Parameters
        ----------
        df : DataFrame -- containing price and trade direction columns
        horizon : int, default=20 -- Number of future candles to check for trade outcome
        sl_mult : float, default=1.0 -- Stop-loss multiplier relative to volatility
        tp_mult : float, default=2.0 -- Take-profit multiplier relative to volatility
        direction_col : str, default="direction" -- Name of the trade direction column (+1=BUY, -1=SELL)
        price_col : str, default="close" -- Name of the entry price column
        high_col : str, default="high" -- Name of the candlestick high column
        low_col : str, default="low" -- Name of the candlestick low column
        vol_col : str, default="volatility_20" -- Name of the volatility column

        Returns
        -------
        outcomes list -- for a new column to DataFram:
            - trade_outcome ∈ {-1, 0, +1, NaN}
            -1 → Stop-loss triggered
            0 → Neither triggered
            +1 → Take-profit triggered
            NaN → Invalid trade direction or insufficient data
        """

        df = df.copy()
        outcomes = []

        for i in range(len(df)):
            if i + horizon >= len(df): outcomes.append(np.nan); continue

            direction = df.loc[i, 'market_regime']
            entry     = df.loc[i, 'close']
            vol       = df.loc[i, 'volatility_20']

            if direction == 1:  # BUY
                TP = entry + tp_mult * vol
                SL = entry - sl_mult * vol

                result = 0
                for j in range(i+1, i+horizon+1):
                    if df.loc[j, 'low' ] <= SL: result = -1; break
                    if df.loc[j, 'high'] >= TP: result =  1; break

            elif direction == -1:  # SELL
                TP = entry - tp_mult * vol
                SL = entry + sl_mult * vol

                result = 0
                for j in range(i+1, i+horizon+1):
                    if df.loc[j, 'high'] >= SL: result = -1; break
                    if df.loc[j, 'low' ] <= TP: result =  1; break

            else: result = np.nan

            outcomes.append(result)
        return outcomes

    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _calc_mfe_mae (df: pd.DataFrame, horizon=20):
        """
        Compute and label MFE and MAE for each hypothetical trade.

        Definitions:
            - MFE (Maximum Favorable Excursion): the maximum positive price movement relative to the entry point
            - MAE (Maximum Adverse Excursion): the maximum negative price movement relative to the entry point

        Formulas:
            MFE = (max(high[t+1:t+horizon]) - entry) / volatility[t]
            MAE = (entry - min(low[t+1:t+horizon])) / volatility[t]

        Parameters
        ----------
        df : DataFrame -- containing price and volatility columns
        horizon : int, default=20 -- Number of future candles used to compute MFE and MAE
        price_col : str, default="close" -- Name of the entry price column
        high_col : str, default="high" -- Name of the candlestick high column
        low_col : str, default="low" -- Name of the candlestick low column
        vol_col : str, default="volatility_20" -- Name of the volatility column

        Returns
        -------
        mfe, mae lists -- for new columns to DataFrame:
            - MFE_norm : normalized MFE value relative to volatility
            - MAE_norm : normalized MAE value relative to volatility
        """

        df = df.copy()

        mfe = []        # Maximum Favorable Excursion (MFE)
        mae = []        # Maximum Adverse Excursion (MAE)

        for i in range(len(df)):
            if i + horizon >= len(df): mfe.append(np.nan); mae.append(np.nan); continue

            entry = df.loc[i, 'close']

            max_high = df.loc[i+1:i+horizon, 'high'].max()
            min_low  = df.loc[i+1:i+horizon, 'low' ].min()

            mfe.append((max_high - entry) / df.loc[i, 'volatility_20'])
            mae.append((entry - min_low ) / df.loc[i, 'volatility_20'])
        return mfe, mae
    

    # ---------------------------------------------------------------------------------------



