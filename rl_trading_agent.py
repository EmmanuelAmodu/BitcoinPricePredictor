import glob
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.mstats import winsorize
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv1D,
                                     Dense, Dropout, Flatten, Input, LSTM)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


def load_klines(path: str = "data/klines") -> pd.DataFrame:
    """Load all CSV klines from a folder into a single DataFrame."""
    files = sorted(glob.glob(str(Path(path) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}")

    frames: List[pd.DataFrame] = []
    for filename in files:
        df = pd.read_csv(filename, header=None)
        df.columns = [
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
        ]
        frames.append(df)

    frame = pd.concat(frames, axis=0, ignore_index=True)
    frame["Open time"] = frame["Open time"].astype("int64")
    frame["Close time"] = frame["Close time"].astype("int64")

    # Normalize timestamp scale: handle ms/us/ns by dividing until within ms range
    while frame["Open time"].max() > 2e13:
        frame["Open time"] //= 1000
        frame["Close time"] //= 1000

    frame["Open time"] = pd.to_datetime(frame["Open time"], unit="ms")
    frame["Close time"] = pd.to_datetime(frame["Close time"], unit="ms")
    return frame


def min_max_scale(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Safe min-max scaling that avoids divide-by-zero for constant columns."""
    for col in columns:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max == col_min:
            df[col] = 0.0
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def engineer_features(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Feature engineering and scaling; returns features and raw closes."""
    # Add time-based features
    time_features = ["Open time", "Close time"]
    for feature in time_features:
        frame[f"{feature} hour"] = frame[feature].dt.hour
        frame[f"{feature} day of week"] = frame[feature].dt.weekday
        frame[f"{feature} week of year"] = frame[feature].dt.isocalendar().week
        frame[f"{feature} month"] = frame[feature].dt.month

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
    ]
    frame[numeric_cols] = frame[numeric_cols].astype(float)

    # Preserve raw closes for reward calculation
    raw_closes = frame["Close"].to_numpy(copy=True)

    # Scale and winsorize to keep extreme outliers in check
    frame = min_max_scale(frame, numeric_cols)
    frame[["Open", "High", "Low", "Close", "Volume"]] = frame[
        ["Open", "High", "Low", "Close", "Volume"]
    ].apply(lambda x: winsorize(x, limits=[0.01, 0.01]))

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Open time hour",
        "Open time day of week",
        "Open time week of year",
        "Open time month",
        "Close time hour",
        "Close time day of week",
        "Close time week of year",
        "Close time month",
    ]
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    return features, raw_closes.astype(np.float32)


def build_windows(
    features: np.ndarray, closes: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows for states and aligned close prices for rewards.
    Returns:
        windows: shape (num_steps, window_size, feature_size)
        close_series: shape (num_steps + 1,) unscaled closes to compute P&L
    """
    if len(features) <= window_size:
        raise ValueError("Not enough rows to build even one window.")
    windows = []
    for start in range(len(features) - window_size):
        windows.append(features[start : start + window_size])

    close_series = closes[window_size - 1 :]
    return np.array(windows, dtype=np.float32), np.array(close_series, dtype=np.float32)


class TradingEnv:
    """Minimal trading environment with hold/long/short and P&L rewards."""

    def __init__(
        self,
        windows: np.ndarray,
        close_series: np.ndarray,
        transaction_cost: float = 0.0005,
    ) -> None:
        """
        Args:
            windows: (steps, window_size, feature_size) prebuilt feature windows
            close_series: (steps + 1,) raw closes aligned with windows
            transaction_cost: flat cost applied when switching position
        """
        if len(close_series) != len(windows) + 1:
            raise ValueError("close_series must be one element longer than windows")
        self.windows = windows
        self.close_series = close_series
        self.window_size = windows.shape[1]
        self.base_feature_size = windows.shape[2]
        self.action_size = 3  # 0: hold, 1: long, 2: short
        self.transaction_cost = transaction_cost
        self.max_idx = len(windows) - 1
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.idx = 0
        self.position = 0  # -1 short, 0 flat, 1 long
        return self._augment_state(self.windows[self.idx])

    def _augment_state(self, state: np.ndarray) -> np.ndarray:
        pos_col = np.full((self.window_size, 1), self.position, dtype=np.float32)
        return np.concatenate([state, pos_col], axis=1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action not in (0, 1, 2):
            raise ValueError("action must be 0 (hold), 1 (long), or 2 (short)")

        prev_position = self.position
        if action == 1:
            self.position = 1
        elif action == 2:
            self.position = -1

        price_now = self.close_series[self.idx]
        price_next = self.close_series[self.idx + 1]
        price_change = price_next - price_now

        reward = self.position * price_change
        if self.position != prev_position:
            reward -= self.transaction_cost

        self.idx += 1
        done = self.idx >= self.max_idx
        next_state = (
            self._augment_state(self.windows[self.idx]) if not done else self._augment_state(self.windows[self.max_idx])
        )
        return next_state, reward, done


class ReplayBuffer:
    def __init__(self, capacity: int = 50000) -> None:
        self.capacity = capacity
        self.buffer: Deque = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in idxs]
        )
        return (
            np.stack(states),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def create_q_model(window_size: int, feature_size: int, action_size: int) -> Model:
    inputs = Input(shape=(window_size, feature_size))

    conv_branch = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Dropout(0.3)(conv_branch)
    conv_branch = Flatten()(conv_branch)

    lstm_branch = LSTM(64, return_sequences=True)(inputs)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Dropout(0.3)(lstm_branch)
    lstm_branch = Flatten()(lstm_branch)

    x = Concatenate()([conv_branch, lstm_branch])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(action_size, activation=None)(x)  # Q-values

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=Huber())
    return model


def train_dqn(
    env: TradingEnv,
    model: Model,
    target_model: Model,
    episodes: int = 5,
    batch_size: int = 32,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    warmup: int = 500,
    target_update: int = 250,
) -> List[float]:
    buffer = ReplayBuffer()
    rewards_history: List[float] = []
    step_count = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_size)
            else:
                q_values = model.predict(state[None, ...], verbose=0)[0]
                action = int(np.argmax(q_values))

            next_state, reward, done = env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            step_count += 1

            if len(buffer) >= warmup:
                (
                    s_batch,
                    a_batch,
                    r_batch,
                    ns_batch,
                    d_batch,
                ) = buffer.sample(batch_size)

                target_q = model.predict(s_batch, verbose=0)
                next_q = target_model.predict(ns_batch, verbose=0)
                max_next_q = np.max(next_q, axis=1)

                updates = r_batch + gamma * max_next_q * (1.0 - d_batch)
                target_q[np.arange(batch_size), a_batch] = updates
                model.train_on_batch(s_batch, target_q)

                if step_count % target_update == 0:
                    target_model.set_weights(model.get_weights())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards_history.append(ep_reward)
        print(f"Episode {ep + 1}: reward={ep_reward:.4f}, epsilon={epsilon:.3f}")

    return rewards_history


def plot_rewards(rewards: List[float]) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training rewards")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_dir = "data/klines"
    if not Path(data_dir).exists():
        raise SystemExit(f"Data directory {data_dir} not found")

    frame = load_klines(data_dir)
    features, closes = engineer_features(frame)

    window_size = 64
    windows, close_series = build_windows(features, closes, window_size)

    # Include position as an extra feature channel
    feature_size = windows.shape[2] + 1
    env = TradingEnv(windows, close_series, transaction_cost=0.0005)

    q_model = create_q_model(window_size, feature_size, env.action_size)
    target_q_model = create_q_model(window_size, feature_size, env.action_size)
    target_q_model.set_weights(q_model.get_weights())

    rewards = train_dqn(
        env,
        q_model,
        target_q_model,
        episodes=3,
        batch_size=32,
        warmup=200,
    )
    plot_rewards(rewards)
