import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.src.models.functional import Functional


# -------------------------------------------------------------------------------------------
class TimeSeriesBranch():
    # ---------------------------------------------------------------------------------------
    def __init__(self, input_shape, filters: list=[32, 64], kernel_size=3, gru_units=32):

        self.input_shape = input_shape
        self.filters     = filters
        self.kernel_size = kernel_size
        self.gru_units   = gru_units

    # ---------------------------------------------------------------------------------------
    def build(self):
        inputs = layers.Input(shape=self.input_shape, name="time_series")   # --> (batch size, step, channels)
        x = inputs

        for i, f in enumerate(self.filters):
            x = layers.Conv1D(f, self.kernel_size, padding="same", activation="relu", dilation_rate=i+1)(x)
            x = layers.BatchNormalization()(x)
        # if i == 0: x = layers.MaxPooling1D(pool_size=2)(x)

        # GRU for sequential dependency
        x = layers.GRU(self.gru_units, return_sequences=True)(x)
        x = layers.GlobalAveragePooling1D()(x)

        return Model(inputs, x, name="time_series_branch")


# -------------------------------------------------------------------------------------------
class CandleBranch():
    # ---------------------------------------------------------------------------------------
    def __init__(self, input_shape, filters: list=[32, 64, 128]):
        self.filters     = filters
        self.input_shape = input_shape

    # ---------------------------------------------------------------------------------------
    def build(self):
        inputs = layers.Input(shape=self.input_shape, name="candle")        # --> (batch size, height, width, channels)
        x = inputs

        for f in self.filters:
            x = layers.Conv2D(f, kernel_size=3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)                  # Added for stability
            x = layers.MaxPooling2D()(x)
            
        x = layers.GlobalAveragePooling2D()(x)
        return Model(inputs, x, name="image_branch")


# -------------------------------------------------------------------------------------------
class ModelCreator():
    # ---------------------------------------------------------------------------------------
    def __init__(self, time_series_arg: list, candle_arg: list, stage_mode: str='entry_probability'):

        self.time_series_shape  = time_series_arg[0]
        self.time_series_filter = time_series_arg[1]
        self.time_series_k_size = time_series_arg[2]
        self.time_series_gru    = time_series_arg[3]

        self.candle_shape       = candle_arg[0]
        self.candle_filter      = candle_arg[1]

        self.stage_mode         = stage_mode
    
        if stage_mode not in ("entry_probability","win_probability"): 
            raise RuntimeError(f"StageMode must be 'entry_probability' or 'win_probability'")

    # ---------------------------------------------------------------------------------------
    def build(self, head_layers: list=[128, 64]):
        branch_input, branch_output = [], []
        
        # TimeSeries branch
        if self.time_series_shape:
            self.feature_branch: Functional = TimeSeriesBranch(self.time_series_shape,
                                                               self.time_series_filter,
                                                               self.time_series_k_size,
                                                               self.time_series_gru).build()
            branch_input .append(self.feature_branch.input )
            branch_output.append(self.feature_branch.output)

        # Image branch
        if self.candle_shape:
            self.candle_branch: Functional = CandleBranch(self.candle_shape, self.candle_filter).build()
            branch_input .append(self.candle_branch.input )
            branch_output.append(self.candle_branch.output)

        # Small processing on auxiliary inputs before merging
        if self.stage_mode == 'win_probability':
            entry_prob_in = layers.Input(shape=(1,), name=self.stage_mode)
            direction_in  = layers.Input(shape=(1,), name="direction")      # +1 buy / -1 sell
            
            aux = layers.Concatenate()([entry_prob_in, direction_in])
            aux = layers.Dense(8, activation="relu")(aux)
            
            branch_input .extend([entry_prob_in, direction_in])
            branch_output.append(aux)


        # Combine all feature outputs
        if len(branch_output) > 1: x = layers.concatenate(branch_output)
        else: x = branch_output[0]
        
        output = self._build_head(x, head_layers)
        self.model = tf.keras.Model(inputs=branch_input, outputs=output, name=self.stage_mode)

        return self.model
    
    # ---------------------------------------------------------------------------------------
    @staticmethod
    def _build_head(x, head_layers: list, dropout=0.3):

        for n in head_layers:
            x = layers.Dense(n, activation="relu")(x)
            
        # x = layers.Dropout(dropout)(x)
        x = layers.Dense(1, activation="sigmoid", name='probability')(x)
        return x




"""
# USAGE EXAMPLE:
# ===========================================================================================

# 1. Build Model with 2-Branch
time_series_arg = [(48, 15), [32, 64], 3, 32]
candle_arg      = [(64, 144, 1), [32, 64, 128]]
model = ModelCreator(time_series_arg, candle_arg, stage_mode='win_probability').build()

# 2. Define metrics & callbacks
metrics_list = [ metrics.AUC(name="auc"), metrics.Precision(name="precision"), metrics.Recall(name="recall") ]
callbackss   = [
        ReduceLROnPlateau(monitor="val_mse", factor=0.5, patience=10, verbose=0, mode="min", min_lr=1e-5),
        EarlyStopping(monitor='val_mse', patience=10, verbose=0, mode="min", start_from_epoch=500),
        ModelCheckpoint(filepath=f"best_model_M.keras", monitor="val_mse", mode="min", verbose=0, save_best_only=True),
        TensorBoard('./', histogram_freq=1, write_graph=False, write_images=False)
        ]

# 3. Run Model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=metrics_list)
model.fit(train_ds, validation_data=valid_ds, epochs=20, callbacks=callbacks)
"""


