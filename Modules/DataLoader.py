import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# market datasets handling windowing, scaling, and candlestick image generation -------------
class BaseMarketDataset(tf.keras.utils.Sequence):
    # ---------------------------------------------------------------------------------------
    def __init__(
        self,
        df: pd.DataFrame, feature_cols: list[str],
        window: int = 48, batch_size: int = 32,
        use_images: bool = True, shuffle: bool = True, indices: np.ndarray | None = None, scaler: StandardScaler | None = None
    ):
        """
        Args:
            df: Input DataFrame.
            feature_cols: List of columns to be used as tabular features.
            window: Number of past time-steps per sample.
            batch_size: Number of samples per batch.
            use_images: Whether to generate and return candlestick images.
            shuffle: Whether to shuffle indices at the start and after each epoch.
            indices: Specific row indices to use. If None, uses all possible windows.
            scaler: A pre-fitted StandardScaler. If None, fits a new one on the provided df.
        """
        super().__init__()

        self.df         = df
        self.window     = window
        self.batch_size = batch_size
        
        self.use_images   = use_images
        self.shuffle      = shuffle
        self.feature_cols = feature_cols

        # If no indices provided, generate all possible start positions
        self.indices = indices if indices is not None else np.arange(len(df) - window)
        
        # If no indices provided, generate all possible start positions
        if scaler is not None: self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            self._fit_scaler()

        if self.shuffle: np.random.shuffle(self.indices)

    # ---------------------------------------------------------------------------------------
    def _fit_scaler(self):
        # Fits the StandardScaler on the tabular features of the current dataframe
        X = self.df[self.feature_cols].values
        self.scaler.fit(X)

    # ---------------------------------------------------------------------------------------
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))    # Total number of batches

    # ---------------------------------------------------------------------------------------
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)        # Randomize data every epoch

    # ---------------------------------------------------------------------------------------
    def candle_image(self, df_window: pd.DataFrame, height: int=64):
        """
        Generate a binary image of candlesticks within a given time window.

        Each candlestick is represented as a column in the image:
        - Wick is drawn as a vertical line
        - Body is drawn as a filled rectangle

        Parameters
        ----------
        df_window : DataFrame -- containing the columns ['open','high','low','close'] for a time window
        height : int, default=64 -- Image height (number of pixels along the y-axis)

        Returns
        -------
        np.ndarray -- Binary array of shape (height, len(df_window)*3) representing the candlestick visualization.
        """
        img = np.zeros((height, len(df_window)*3))
        norm = self._normalize_window(df_window)

        for i, row in norm.iterrows():
            x = (i - df_window.index[0]) * 3 + 1

            o, h, l, c = row
            y_o = int(o * (height-1))
            y_c = int(c * (height-1))
            y_h = int(h * (height-1))
            y_l = int(l * (height-1))

            img[y_l:y_h+1, x] = 1							# wick
            img[min(y_o,y_c):max(y_o,y_c)+1, x-1:x+2] = 1	# body

        return img
    
    # p′ = (p−min(low)) / (max(high)−min(low)) ----------------------------------------------
    @staticmethod
    def _normalize_window(df_window: pd.DataFrame):
        low  = df_window['low' ].min()
        high = df_window['high'].max()
        return (df_window[['open','high','low','close']] - low) / (high - low)


# Market Regime Classification & Trade Outcome Prediction -----------------------------------
class MarketDataset(BaseMarketDataset):
    def __init__(self, stage_mode: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stage_mode = stage_mode
        if stage_mode == 'win_probability':
            # Filter indices where regime is not 0 and trade outcome is valid (-1 or 1)
            valid_1 = self.df["market_regime"].values[self.indices + self.window] != 0
            valid_2 = np.isin(self.df["trade_outcome"].values[self.indices + self.window], [-1, 1])
            self.indices = self.indices[valid_1 & valid_2]

            # Precalculation of auxiliary inputs for win probability stage
            self._entry_probs = self.df.iloc[self.indices + self.window]["entry_probability"].values.astype(np.float32)
            self._directions  = self.df.iloc[self.indices + self.window]["market_regime"]    .values.astype(np.float32)

            # Pre-calculate targets for efficiency - 'trade_outcome' (mapped -1 -> 0, 1 -> 1)
            self._targets = (self.df.iloc[self.indices + self.window]["trade_outcome"].map({-1: 0, 1: 1}).values.astype(np.int32))
            
        elif stage_mode == 'entry_probability': 
            self._targets = self.df.iloc[self.indices + self.window]["market_regime"].values.astype(np.int32)
        else:
            raise RuntimeError(f"StageMode must be 'entry_probability' or 'win_probability'")

    # ---------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        start, end = idx*self.batch_size, (idx+1)*self.batch_size
        batch_indices = self.indices [start:end]
        batch_targrts = self._targets[start:end]

        feature_input, candles_input = [], []

        for i in batch_indices:
            window_df = self.df.iloc[i:i+self.window]
            feature_input.append(self.scaler.transform(window_df[self.feature_cols].values))
            if self.use_images: candles_input.append(self.candle_image(window_df))

        inputs = [np.array(feature_input, dtype=np.float32)]
        if self.use_images: inputs.append(np.array(candles_input, dtype=np.float32))
        if self.stage_mode == 'win_probability':
            inputs.append(self._entry_probs[start:end])
            inputs.append(self._directions [start:end])
        
        inputs  = tuple(inputs) if len(inputs) > 1 else tuple(inputs[0])
        outputs = np.array(batch_targrts, dtype=np.int32)

        return inputs, outputs


# Splits the dataframe into training, validation, and test indices for time-series ----------
class DatasetManager():
    # ---------------------------------------------------------------------------------------
    def __init__(self, df: pd.DataFrame, window: int = 48, split_ratio: tuple=(0.7, 0.15, 0.15)):
        self.df = df.reset_index(drop=True)
        self.window = window
        self.split_ratio = split_ratio
        self._split()

    # ---------------------------------------------------------------------------------------
    def _split(self):
        n_samples = len(self.df) - self.window

        n_train = int(self.split_ratio[0] * n_samples)
        n_val   = int(self.split_ratio[1] * n_samples)

        self.train_idx = np.arange(0, n_train)
        self.val_idx   = np.arange(n_train, n_train + n_val)
        self.test_idx  = np.arange(n_train + n_val, n_samples)



"""
# USAGE EXAMPLE:
# ===========================================================================================

# 1. Load and Clean Data
data = pd.read_csv('your_file.csv').reset_index(drop=True)
feature_cols = [
    "time_sin", "time_cos",
    "return", "body", "range", "upper_wick", "lower_wick",
    "ema_20", "price_ema_ratio", "volatility_20", "volume_norm",
    "dist_round", "trend_144", "dist_shadow_prev_time", "dist_shadow_prev_week"
]

# 2. Initialize Splitter
manager = DatasetManager(data, window=48)

# 3. Create TRAIN dataset (this will FIT the scaler)
train_ds = MarketDataset('win_probability', data, feature_cols, indices=manager.train_idx, shuffle=True)
shared_scaler = train_ds.scaler                     # Capture the fitted scaler

# 4. Create VAL/TEST datasets (passing the SHARED scaler - only TRANSFORM will happen)
valid_ds = MarketDataset('win_probability', data, feature_cols, scaler=shared_scaler, indices=manager.val_idx , shuffle=False)
test_ds  = MarketDataset('win_probability', data, feature_cols, scaler=shared_scaler, indices=manager.test_idx, shuffle=False)
"""
