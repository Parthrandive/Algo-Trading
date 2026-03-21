from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence


# To add a new stock only change this one block - nothing else:
EQUITY_SYMBOLS = ["INFY.NS", "RELIANCE.NS", "TATASTEEL.NS", "TCS.NS"]
# Add any NSE symbol here:
# EQUITY_SYMBOLS += ["HDFCBANK.NS", "WIPRO.NS"]

FOREX_SYMBOLS = ["USDINR=X"]
# USDINR is never a prediction target - external feature only

TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"
FOCAL_ALPHA = [0.25, 0.50, 0.25]  # [down, neutral, up]
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 128


def _to_datetime_index(df: pd.DataFrame, name: str, require_volume: bool) -> pd.DataFrame:
    """Validate and normalize an already-loaded market frame.

    No loading logic is introduced here; this only validates/cleans the provided frame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{name}] expected DataFrame, got {type(df)}")

    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"[{name}] index must be DatetimeIndex")

    frame = frame.sort_index()
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}")

    if require_volume and "volume" not in frame.columns:
        raise ValueError(f"[{name}] equity frame must include volume")

    if frame["close"].isna().any():
        raise ValueError(f"[{name}] close has missing values")

    return frame


def _normalize_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is not None:
        return index.tz_convert(None).normalize()
    return index.normalize()


def prepare_usdinr_features(df_usdinr: pd.DataFrame) -> pd.DataFrame:
    df_usdinr = _to_datetime_index(df_usdinr, "USDINR", require_volume=False)
    df_usdinr = df_usdinr.drop(columns=["volume"], errors="ignore")

    df_usdinr["usdinr_return"] = np.log(df_usdinr["close"] / df_usdinr["close"].shift(1))
    df_usdinr["usdinr_zscore"] = (
        (df_usdinr["usdinr_return"] - df_usdinr["usdinr_return"].rolling(20).mean())
        / (df_usdinr["usdinr_return"].rolling(20).std() + 1e-9)
    )
    df_usdinr["usdinr_trend"] = df_usdinr["close"] / df_usdinr["close"].rolling(20).mean() - 1
    df_usdinr["usdinr_vol"] = df_usdinr["usdinr_return"].rolling(20).std()

    df_usdinr = df_usdinr.rename(columns={"close": "usdinr_close"})[
        ["usdinr_close", "usdinr_return", "usdinr_zscore", "usdinr_trend", "usdinr_vol"]
    ].dropna()

    print(
        f"[USDINR] Ready: {len(df_usdinr)} rows | "
        f"{df_usdinr.index[0].date()} -> {df_usdinr.index[-1].date()}"
    )
    return df_usdinr


def merge_usdinr_features(df: pd.DataFrame, df_usdinr_feat: pd.DataFrame, symbol: str) -> pd.DataFrame:
    usd_cols = ["usdinr_close", "usdinr_return", "usdinr_zscore", "usdinr_trend", "usdinr_vol"]
    # Align forex context to equity timestamps using historical forward-fill semantics.
    aligned = (
        df_usdinr_feat[usd_cols]
        .reindex(df.index.union(df_usdinr_feat.index))
        .sort_index()
        .ffill()
        .reindex(df.index)
    )
    df = df.join(aligned, how="left")
    for col in usd_cols:
        df[col] = df[col].ffill()
    print(f"  [{symbol}] Rows after USDINR merge: {len(df)}")
    return df


def engineer_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    window = max(5, min(20, len(df) // 10))
    print(f"  [{symbol}] Feature window: {window}")

    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["sma"] = df["close"].rolling(window).mean()
    df["ema"] = df["close"].ewm(span=window).mean()
    df["sma_ratio"] = df["close"] / (df["sma"] + 1e-9)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    bb_mid = df["close"].rolling(window).mean()
    bb_std = df["close"].rolling(window).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df["volume_zscore"] = (
        (df["volume"] - df["volume"].rolling(window).mean())
        / (df["volume"].rolling(window).std() + 1e-9)
    )
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(window).mean() + 1e-9)

    df["vol_20"] = df["log_return"].rolling(window).std()
    try:
        df["vol_regime"] = pd.qcut(df["vol_20"], 3, labels=[0, 1, 2], duplicates="drop").astype(float)
    except ValueError:
        # Fallback when qcut cannot produce stable bins for small/flat samples.
        df["vol_regime"] = 1.0

    df["return_zscore"] = (
        (df["log_return"] - df["log_return"].rolling(window).mean())
        / (df["log_return"].rolling(window).std() + 1e-9)
    )

    for lag in [1, 2, 3, 5]:
        df[f"return_lag{lag}"] = df["log_return"].shift(lag)
        df[f"rsi_lag{lag}"] = df["rsi"].shift(lag)

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        print(f"  [{symbol}] Nulls before dropna: {nulls.to_dict()}")

    df = df.dropna()
    print(f"  [{symbol}] Rows after features: {len(df)}")

    assert len(df) >= 300, (
        f"[{symbol}] Only {len(df)} rows after features - check data pipeline"
    )
    return df


def find_threshold(
    train_log_returns: np.ndarray,
    symbol: str,
    target_min: float = 0.20,
    target_max: float = 0.25,
    min_thresh: float = 0.0020,
    max_thresh: float = 0.0200,
    step: float = 0.0002,
) -> float:
    best_thresh, best_diff = min_thresh, 999.0

    for thresh in np.arange(min_thresh, max_thresh, step):
        thresh = round(float(thresh), 6)
        labels = np.where(train_log_returns > thresh, 2, np.where(train_log_returns < -thresh, 0, 1))
        neutral_pct = (labels == 1).mean()

        if target_min <= neutral_pct <= target_max:
            print(f"  [{symbol}] Threshold {thresh:.4f} -> neutral={neutral_pct:.1%} OK")
            return thresh

        diff = abs(neutral_pct - 0.22)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh

    fallback_labels = np.where(
        train_log_returns > best_thresh,
        2,
        np.where(train_log_returns < -best_thresh, 0, 1),
    )
    fallback_neutral = (fallback_labels == 1).mean()
    print(f"  [{symbol}] Fallback threshold {best_thresh:.4f} -> neutral={fallback_neutral:.1%}")
    return best_thresh


def apply_labels(log_returns: np.ndarray, thresh: float) -> np.ndarray:
    return np.where(log_returns > thresh, 2, np.where(log_returns < -thresh, 0, 1))


def compute_gamma(y_train: np.ndarray, symbol: str) -> float:
    counts = np.bincount(y_train.astype(int), minlength=3)
    dominant = counts.max()
    non_zero = counts[counts > 0]
    minority = non_zero.min() if len(non_zero) else 1
    imbalance = dominant / (minority + 1e-6)

    if imbalance > 6.0:
        gamma = 5.0
    elif imbalance > 4.0:
        gamma = 4.0
    elif imbalance > 2.5:
        gamma = 3.0
    elif imbalance > 1.5:
        gamma = 2.0
    else:
        gamma = 1.5

    print(f"  [{symbol}] Counts: {counts} | Imbalance: {imbalance:.2f} | Gamma: {gamma}")
    return gamma


def focal_loss(gamma: float = 2.0, alpha: list[float] | tuple[float, float, float] = FOCAL_ALPHA):
    alpha_vec = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=3)
        y_pred_clip = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred_clip), axis=-1)
        pt = tf.reduce_sum(y_true_oh * y_pred_clip, axis=-1)
        alpha_t = tf.reduce_sum(y_true_oh * alpha_vec, axis=-1)
        focal_w = alpha_t * tf.pow(1.0 - pt, gamma)
        return tf.reduce_mean(focal_w * ce)

    return loss_fn


def _compute_class_weight_dict(y: np.ndarray) -> dict[int, float]:
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    weights = {int(cls): float(weight) for cls, weight in zip(classes, cw)}
    # Asymmetric penalty: amplify missed down moves.
    weights[0] = float(weights.get(0, 1.0) * 2.0)
    return weights


class NeutralFloorMonitor(Callback):
    def __init__(
        self,
        symbol: str,
        X_val: np.ndarray,
        check_every: int = 10,
        neutral_floor: float = 0.03,
    ) -> None:
        super().__init__()
        self.symbol = symbol
        self.X_val = X_val
        self.check_every = int(check_every)
        self.neutral_floor = float(neutral_floor)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        epoch_num = int(epoch) + 1
        if epoch_num % self.check_every != 0:
            return
        preds = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        if len(preds) == 0:
            return
        neutral_rate = float((preds == 1).mean())
        if neutral_rate < self.neutral_floor:
            print(
                f"WARNING: [{self.symbol}] Epoch {epoch_num}: neutral collapsed "
                f"({neutral_rate:.2%}) - focal loss alpha may need adjustment"
            )


class BalancedBatchGenerator(Sequence):
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = DEFAULT_BATCH_SIZE):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.classes = np.unique(y)
        if len(self.classes) == 0:
            raise ValueError("BalancedBatchGenerator received empty label array")

        base = batch_size // len(self.classes)
        remainder = batch_size % len(self.classes)
        self.samples_per_class = {
            int(c): base + (idx < remainder) for idx, c in enumerate(self.classes)
        }

        self.indices_per_class = {int(c): np.where(y == c)[0] for c in self.classes}
        for c, idx in self.indices_per_class.items():
            if len(idx) < 10:
                print(f"  WARNING: class {c} has only {len(idx)} samples")

    def __len__(self) -> int:
        return max(len(self.y) // self.batch_size, 1)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_idx: list[int] = []
        for c in self.classes:
            c_int = int(c)
            n_pick = self.samples_per_class[c_int]
            chosen = np.random.choice(self.indices_per_class[c_int], n_pick, replace=True)
            batch_idx.extend(chosen.tolist())
        np.random.shuffle(batch_idx)
        batch_idx_arr = np.asarray(batch_idx, dtype=int)
        return self.X[batch_idx_arr], self.y[batch_idx_arr]


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])

    if not Xs:
        raise ValueError(f"Not enough rows to create sequences (seq_len={seq_len}, rows={len(X)})")

    return np.asarray(Xs), np.asarray(ys)


def build_cnn(input_shape: tuple[int, int], gamma: float = 2.0, shallow: bool = False, symbol: str = ""):
    if shallow:
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv1D(32, kernel_size=3, activation="relu"),
                BatchNormalization(),
                Flatten(),
                Dense(64, activation="relu", kernel_regularizer=l2(1e-3)),
                Dropout(0.5),
                Dense(3, activation="softmax"),
            ]
        )
    else:
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv1D(64, kernel_size=3, activation="relu"),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Conv1D(128, kernel_size=3, activation="relu"),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
                Dropout(0.4),
                Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
                Dropout(0.3),
                Dense(3, activation="softmax"),
            ]
        )

    model.compile(
        optimizer=Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss=focal_loss(gamma=gamma, alpha=FOCAL_ALPHA),
        metrics=["accuracy"],
    )
    arch = "shallow" if shallow else "full"
    print(f"  [{symbol}] CNN built: {arch} | gamma={gamma}")
    return model


def check_collapse(preds: np.ndarray, symbol: str, threshold: float = 0.05) -> bool:
    counts = np.bincount(preds, minlength=3)
    freqs = counts / (counts.sum() + 1e-9)
    labels = {0: "down", 1: "neutral", 2: "up"}
    collapsed = [labels[i] for i, freq in enumerate(freqs) if freq < threshold]
    if collapsed:
        print(f"  [{symbol}] Collapse in: {collapsed}")
        print(f"  [{symbol}] Pred dist -> down:{counts[0]} neutral:{counts[1]} up:{counts[2]}")
    return len(collapsed) > 0


def get_callbacks(symbol: str, model_dir: Path, X_val_seq: np.ndarray | None = None):
    safe = symbol.replace(".", "_").replace("=", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[Callback] = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        ModelCheckpoint(model_dir / f"best_cnn_{safe}.keras", monitor="val_loss", save_best_only=True),
    ]
    if X_val_seq is not None:
        callbacks.append(NeutralFloorMonitor(symbol=symbol, X_val=X_val_seq))
    return callbacks


def train_with_retry(
    symbol: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_v: np.ndarray,
    y_v: np.ndarray,
    gamma: float,
    class_weight_dict: dict[int, float],
    model_dir: Path,
    seq_len: int = 30,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = 150,
    verbose: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float, int]:
    X_tr_s, y_tr_s = make_sequences(X_tr, y_tr, seq_len)
    X_v_s, y_v_s = make_sequences(X_v, y_v, seq_len)
    input_shape = (seq_len, X_tr.shape[1])

    model = build_cnn(input_shape, gamma=gamma, shallow=False, symbol=symbol)
    gen = BalancedBatchGenerator(X_tr_s, y_tr_s, batch_size=batch_size)
    history = model.fit(
        gen,
        validation_data=(X_v_s, y_v_s),
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=get_callbacks(symbol, model_dir, X_val_seq=X_v_s),
        verbose=verbose,
    )
    print(f"  [{symbol}] Attempt 1: {len(history.history['loss'])} epochs")

    val_preds = np.argmax(model.predict(X_v_s, verbose=0), axis=1)

    architecture = "full"
    gamma_final = gamma
    epochs_trained = len(history.history["loss"])
    if check_collapse(val_preds, symbol):
        gamma_retry = min(gamma + 2.0, 5.0)
        print(f"  [{symbol}] Retrying: shallow=True gamma={gamma_retry}")

        model = build_cnn(input_shape, gamma=gamma_retry, shallow=True, symbol=symbol)
        gen = BalancedBatchGenerator(X_tr_s, y_tr_s, batch_size=batch_size)
        history = model.fit(
            gen,
            validation_data=(X_v_s, y_v_s),
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=get_callbacks(symbol, model_dir, X_val_seq=X_v_s),
            verbose=verbose,
        )
        print(f"  [{symbol}] Attempt 2: {len(history.history['loss'])} epochs")
        architecture = "shallow"
        gamma_final = gamma_retry
        epochs_trained = len(history.history["loss"])

    return model, X_tr_s, y_tr_s, X_v_s, y_v_s, architecture, gamma_final, epochs_trained


def get_n_splits(n_samples: int, symbol: str, min_train: int = 200, min_val: int = 50) -> int:
    for n in [5, 4, 3, 2]:
        fold_size = n_samples // (n + 1)
        if fold_size >= min_val and n_samples - fold_size >= min_train:
            print(f"  [{symbol}] Using {n}-fold CV")
            return n
    print(f"  WARNING: [{symbol}] too few samples - single split fallback")
    return 1


def run_walk_forward_cv(
    symbol: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seq_len: int,
    batch_size: int,
    epochs: int,
    verbose: int,
) -> tuple[float, float]:
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    n_splits = get_n_splits(len(X_full), symbol)
    fold_scores: list[float] = []

    if n_splits == 1:
        split_at = max(200, len(X_full) - 50)
        split_at = min(split_at, len(X_full) - 1)
        split_iterator = [(np.arange(0, split_at), np.arange(split_at, len(X_full)))]
    else:
        split_iterator = TimeSeriesSplit(n_splits=n_splits).split(X_full)

    for fold, (tr_idx, v_idx) in enumerate(split_iterator, start=1):
        X_tr_f = X_full[tr_idx]
        X_v_f = X_full[v_idx]
        y_tr_f = y_full[tr_idx]
        y_v_f = y_full[v_idx]

        sc_f = StandardScaler()
        X_tr_f = sc_f.fit_transform(X_tr_f)
        X_v_f = sc_f.transform(X_v_f)

        cw_f = _compute_class_weight_dict(y_tr_f)
        gam_f = compute_gamma(y_tr_f, symbol)

        try:
            X_tf_s, y_tf_s = make_sequences(X_tr_f, y_tr_f, seq_len)
            X_vf_s, y_vf_s = make_sequences(X_v_f, y_v_f, seq_len)
        except ValueError:
            continue

        m_f = build_cnn((seq_len, X_tr_f.shape[1]), gamma=gam_f, shallow=False, symbol=symbol)
        gen_f = BalancedBatchGenerator(X_tf_s, y_tf_s, batch_size=batch_size)
        m_f.fit(
            gen_f,
            validation_data=(X_vf_s, y_vf_s),
            epochs=epochs,
            class_weight=cw_f,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                NeutralFloorMonitor(symbol=f"{symbol}-CV{fold}", X_val=X_vf_s),
            ],
            verbose=verbose,
        )

        preds_f = np.argmax(m_f.predict(X_vf_s, verbose=0), axis=1)
        bal_f = balanced_accuracy_score(y_vf_s, preds_f)
        fold_scores.append(float(bal_f))
        print(f"  [{symbol}] Fold {fold} Bal Acc: {bal_f:.4f}")

    if not fold_scores:
        print(f"  WARNING: [{symbol}] no valid CV folds; returning NaN")
        return float("nan"), float("nan")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    print(f"  [{symbol}] CV: {cv_mean:.4f} +/- {cv_std:.4f}")

    if cv_std > 0.05:
        print(f"  WARNING: [{symbol}] high CV variance - extend data history or increase dropout")
    if cv_mean < 0.38:
        print(f"  WARNING: [{symbol}] low CV mean - check threshold and focal loss settings")

    return cv_mean, cv_std


def _select_feature_columns(train_df: pd.DataFrame) -> list[str]:
    excluded = {"open", "high", "low", "close", "volume", "log_return"}
    cols: list[str] = []
    for col in train_df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns available after exclusions")
    return cols


def print_validation_checklist(
    symbol: str,
    symbol_results: dict[str, dict[str, Any]],
    neutral_precision: float,
    neutral_recall: float,
    neutral_f1: float,
    down_recall: float,
    epochs_trained: int,
    val_bal: float,
    test_bal: float,
    gap: float,
    cv_std: float,
) -> None:
    forex_in_results = any(sym in FOREX_SYMBOLS or sym.endswith("=X") for sym in symbol_results.keys())
    checks = [
        ("USDINR in results table", "Never", "Yes" if forex_in_results else "No", not forex_in_results),
        (
            "Neutral P/R/F1 per symbol",
            "All > 0.0",
            f"{neutral_precision:.3f}/{neutral_recall:.3f}/{neutral_f1:.3f}",
            neutral_precision > 0.0 and neutral_recall > 0.0 and neutral_f1 > 0.0,
        ),
        ("Down recall per symbol", "> 0.10", f"{down_recall:.3f}", down_recall > 0.10),
        ("Epochs trained", "< 150", str(epochs_trained), epochs_trained < 150),
        ("Val Balanced Acc", "> 0.42", f"{val_bal:.4f}", val_bal > 0.42),
        ("Test Balanced Acc", "> 0.40", f"{test_bal:.4f}", test_bal > 0.40),
        ("Val->Test gap", "< 0.08", f"{gap:.4f}", gap < 0.08),
        ("CV Std", "< 0.05", f"{cv_std:.4f}", np.isfinite(cv_std) and cv_std < 0.05),
    ]

    print(f"\n  [{symbol}] VALIDATION CHECKLIST")
    print("  Metric                          | Target       | Actual")
    print("  ---------------------------------------------------------------")
    for metric, target, actual, ok in checks:
        print(f"  {metric:<31} | {target:<12} | {actual}")
        if not ok:
            print(f"  WARNING: [{symbol}] {metric} outside target")


def run_multi_symbol_training(
    equity_frames: dict[str, pd.DataFrame],
    df_usdinr: pd.DataFrame,
    seq_len: int = 30,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = 150,
    verbose: int = 0,
    model_dir: str = "models/multi_symbol_cnn",
) -> dict[str, Any]:
    """Train one CNN per equity symbol using preloaded DataFrames only.

    Parameters
    ----------
    equity_frames
        Mapping: symbol -> raw OHLCV DataFrame with DatetimeIndex.
    df_usdinr
        Raw OHLC DataFrame with DatetimeIndex for USDINR external features.
    """

    SYMBOL_CONFIGS: dict[str, dict[str, Any]] = {}
    SYMBOL_MODELS: dict[str, Any] = {}
    SYMBOL_SCALERS: dict[str, StandardScaler] = {}
    SYMBOL_RESULTS: dict[str, dict[str, Any]] = {}

    tf.random.set_seed(42)
    np.random.seed(42)

    invalid_equity_symbols = [sym for sym in EQUITY_SYMBOLS if sym in FOREX_SYMBOLS or sym.endswith("=X")]
    assert not invalid_equity_symbols, (
        f"EQUITY_SYMBOLS contains forex symbol(s): {invalid_equity_symbols}. "
        "USDINR must be external feature only."
    )

    df_usdinr_feat = prepare_usdinr_features(df_usdinr)

    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    for symbol in EQUITY_SYMBOLS:
        print(f"\n{'='*60}")
        print(f"Processing: {symbol}")
        print(f"{'='*60}")

        if symbol not in equity_frames:
            print(f"  WARNING: [{symbol}] missing in equity_frames - skipped")
            continue

        df = _to_datetime_index(equity_frames[symbol], symbol, require_volume=True)

        df = merge_usdinr_features(df, df_usdinr_feat, symbol)
        all_null_cols = [col for col in df.columns if df[col].isna().all()]
        if all_null_cols:
            print(f"  [{symbol}] Dropping all-null columns before features: {all_null_cols}")
            df = df.drop(columns=all_null_cols)
        df = engineer_features(df, symbol)

        norm_idx = _normalize_dates(df.index)
        train_df = df[norm_idx <= TRAIN_END]
        val_df = df[(norm_idx > TRAIN_END) & (norm_idx <= VAL_END)]
        test_df = df[norm_idx > VAL_END]

        print(f"  [{symbol}] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        if len(train_df) < 200:
            print(f"  WARNING: [{symbol}] train too small ({len(train_df)}) - check data pipeline")
        if len(val_df) < 50:
            print(f"  WARNING: [{symbol}] val too small ({len(val_df)}) - check date range")

        thresh = find_threshold(train_df["log_return"].dropna().to_numpy(), symbol)
        SYMBOL_CONFIGS[symbol] = {"threshold": float(thresh)}

        y_train = apply_labels(train_df["log_return"].shift(-1).dropna().to_numpy(), thresh)
        y_val = apply_labels(val_df["log_return"].shift(-1).dropna().to_numpy(), thresh)
        y_test = apply_labels(test_df["log_return"].shift(-1).dropna().to_numpy(), thresh)

        for split_name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
            if len(labels) == 0:
                print(f"  WARNING: [{symbol}] {split_name} labels empty")
                continue
            up = (labels == 2).mean()
            neu = (labels == 1).mean()
            dn = (labels == 0).mean()
            print(f"  [{symbol}] {split_name}: up={up:.1%} neutral={neu:.1%} down={dn:.1%}")
            if neu > 0.40:
                print(f"  WARNING: [{symbol}] neutral still high in {split_name} - threshold needs review")
            if up < 0.20 or dn < 0.20:
                print(f"  WARNING: [{symbol}] up/down too small in {split_name} - widen threshold")

        feature_cols = _select_feature_columns(train_df)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].iloc[:-1])
        X_val = scaler.transform(val_df[feature_cols].iloc[:-1])
        X_test = scaler.transform(test_df[feature_cols].iloc[:-1])
        SYMBOL_SCALERS[symbol] = scaler

        class_weight_dict = _compute_class_weight_dict(y_train)
        print(f"  [{symbol}] Asymmetric class weights (down x2): {class_weight_dict}")

        gamma = compute_gamma(y_train, symbol)
        SYMBOL_CONFIGS[symbol]["gamma"] = float(gamma)

        symbol_model_dir = model_root / symbol.replace(".", "_").replace("=", "_")
        model, X_tr_s, y_tr_s, X_v_s, y_v_s, architecture, gamma_final, epochs_trained = train_with_retry(
            symbol=symbol,
            X_tr=X_train,
            y_tr=y_train,
            X_v=X_val,
            y_v=y_val,
            gamma=gamma,
            class_weight_dict=class_weight_dict,
            model_dir=symbol_model_dir,
            seq_len=seq_len,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
        )
        SYMBOL_MODELS[symbol] = model
        SYMBOL_CONFIGS[symbol]["architecture"] = architecture
        SYMBOL_CONFIGS[symbol]["gamma_final"] = float(gamma_final)

        cv_mean, cv_std = run_walk_forward_cv(
            symbol=symbol,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seq_len=seq_len,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
        )

        X_te_s, y_te_s = make_sequences(X_test, y_test, seq_len)
        val_preds = np.argmax(model.predict(X_v_s, verbose=0), axis=1)
        test_preds = np.argmax(model.predict(X_te_s, verbose=0), axis=1)

        val_bal = float(balanced_accuracy_score(y_v_s, val_preds))
        test_bal = float(balanced_accuracy_score(y_te_s, test_preds))
        gap = abs(val_bal - test_bal)

        v_pred_c = np.bincount(val_preds, minlength=3)
        v_true_c = np.bincount(y_v_s, minlength=3)

        print(f"\n  [{symbol}] -- RESULTS --")
        print(f"  Val Balanced Acc:  {val_bal:.4f}")
        print(f"  Test Balanced Acc: {test_bal:.4f}")
        print(f"  Val->Test Gap:     {gap:.4f}")
        print(f"  Val pred  -> up:{v_pred_c[2]} neutral:{v_pred_c[1]} down:{v_pred_c[0]}")
        print(f"  Val actual -> up:{v_true_c[2]} neutral:{v_true_c[1]} down:{v_true_c[0]}")
        val_report = classification_report(
            y_v_s,
            val_preds,
            labels=[0, 1, 2],
            target_names=["down", "neutral", "up"],
            output_dict=True,
            zero_division=0,
        )
        print(
            classification_report(
                y_v_s,
                val_preds,
                labels=[0, 1, 2],
                target_names=["down", "neutral", "up"],
                zero_division=0,
            )
        )

        SYMBOL_RESULTS[symbol] = {
            "threshold": float(thresh),
            "gamma": float(gamma_final),
            "architecture": architecture,
            "epochs_trained": int(epochs_trained),
            "neutral_pct": float((y_train == 1).mean()) if len(y_train) else float("nan"),
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "val_bal_acc": val_bal,
            "test_bal_acc": test_bal,
            "val_test_gap": gap,
            "neutral_precision_val": float(val_report["neutral"]["precision"]),
            "neutral_recall_val": float(val_report["neutral"]["recall"]),
            "neutral_f1_val": float(val_report["neutral"]["f1-score"]),
            "down_recall_val": float(val_report["down"]["recall"]),
        }

        print_validation_checklist(
            symbol=symbol,
            symbol_results=SYMBOL_RESULTS,
            neutral_precision=float(val_report["neutral"]["precision"]),
            neutral_recall=float(val_report["neutral"]["recall"]),
            neutral_f1=float(val_report["neutral"]["f1-score"]),
            down_recall=float(val_report["down"]["recall"]),
            epochs_trained=int(epochs_trained),
            val_bal=val_bal,
            test_bal=test_bal,
            gap=gap,
            cv_std=cv_std,
        )

    print("\n" + "=" * 90)
    print("CROSS-SYMBOL SUMMARY")
    print("=" * 90)
    print(
        f"{'Symbol':<16} {'Thresh':>8} {'Gamma':>6} {'Arch':>8} {'Neutral%':>9} "
        f"{'CV Mean':>8} {'CV Std':>7} {'Val Bal':>8} {'Test Bal':>9} {'Gap':>6} {'Status':>7}"
    )
    print("-" * 90)

    for sym in EQUITY_SYMBOLS:
        if sym not in SYMBOL_RESULTS:
            continue
        r = SYMBOL_RESULTS[sym]
        status = "OK" if (
            np.isfinite(r["cv_mean"])
            and r["cv_mean"] >= 0.38
            and r["cv_std"] <= 0.05
            and r["test_bal_acc"] >= 0.38
            and r["val_test_gap"] <= 0.08
        ) else "WARN"

        print(
            f"{sym:<16} "
            f"{r['threshold']:>8.4f} "
            f"{r['gamma']:>6.1f} "
            f"{r['architecture']:>8} "
            f"{r['neutral_pct']:>9.1%} "
            f"{r['cv_mean']:>8.4f} "
            f"{r['cv_std']:>7.4f} "
            f"{r['val_bal_acc']:>8.4f} "
            f"{r['test_bal_acc']:>9.4f} "
            f"{r['val_test_gap']:>6.4f} "
            f"{status:>7}"
        )

    print("\n[USDINR=X] Used as external feature only - not a prediction target.")

    for sym in EQUITY_SYMBOLS:
        if sym not in SYMBOL_RESULTS:
            continue
        r = SYMBOL_RESULTS[sym]
        if r["cv_mean"] < 0.38:
            print(f"WARNING: [{sym}] CV mean {r['cv_mean']:.4f} < 0.38 - check threshold + focal loss")
        if r["cv_std"] > 0.05:
            print(f"WARNING: [{sym}] CV std {r['cv_std']:.4f} > 0.05 - add data history or dropout")
        if r["test_bal_acc"] < 0.38:
            print(f"WARNING: [{sym}] Test bal {r['test_bal_acc']:.4f} below target")
        if r["val_test_gap"] > 0.08:
            print(f"WARNING: [{sym}] Val->Test gap {r['val_test_gap']:.4f} - overfitting, extend data history")

    return {
        "SYMBOL_CONFIGS": SYMBOL_CONFIGS,
        "SYMBOL_MODELS": SYMBOL_MODELS,
        "SYMBOL_SCALERS": SYMBOL_SCALERS,
        "SYMBOL_RESULTS": SYMBOL_RESULTS,
    }


if __name__ == "__main__":
    raise SystemExit(
        "This script expects preloaded DataFrames from the existing pipeline. "
        "Call run_multi_symbol_training(equity_frames, df_usdinr)."
    )
