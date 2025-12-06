import glob
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.mstats import winsorize
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv1D,
                                     Dense, Dropout, Flatten, Input, LSTM,
                                     Lambda)
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision


logger = logging.getLogger(__name__)


def setup_gpu():
    """Configure TensorFlow for optimal GPU/Metal performance."""
    logger.info("Setting up GPU/accelerator configuration...")
    
    # List available devices
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Available GPUs: {gpus}")
    
    has_gpu = len(gpus) > 0

    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU memory growth setup failed: {e}")
    else:
        # Check for Metal (Apple Silicon)
        logger.info("No CUDA GPUs found. Checking for Metal acceleration...")
        all_devices = tf.config.list_physical_devices()
        logger.info(f"All available devices: {all_devices}")
        # Note: tf-metal reports as GPU; no special handling needed here.
    
    # Enable mixed precision for faster training on supported hardware
    if has_gpu:
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            logger.info(f"Mixed precision enabled: {policy.name}")
            logger.info(f"Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}")
        except Exception as e:
            logger.warning(f"Mixed precision setup failed (will use float32): {e}")
    else:
        logger.info("Mixed precision not enabled (no GPU/Metal detected)")
    
    # Enable XLA compilation for better performance (only if GPU present)
    if has_gpu:
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("XLA JIT compilation enabled")
        except Exception as e:
            logger.warning(f"XLA JIT enable failed: {e}")
    else:
        logger.info("XLA JIT not enabled (no GPU/Metal detected)")
    
    return has_gpu


def load_klines(path: str = "data/klines") -> pd.DataFrame:
    """Load all CSV klines from a folder into a single DataFrame."""
    files = sorted(glob.glob(str(Path(path) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}")
    logger.info("Loading %d kline files from %s", len(files), path)

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
    logger.info("Loaded %d rows of klines", len(frame))
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
    logger.info("Engineering features for %d rows", len(frame))
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
    logger.info(
        "Feature matrix built with shape %s and close series length %d",
        features.shape,
        raw_closes.shape[0],
    )
    return features, raw_closes.astype(np.float32)


def build_windows(
    features: np.ndarray, closes: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows for states and aligned close prices for rewards.
    Returns:
        windows: shape (num_steps, window_size, feature_size)
        close_series: shape (num_steps + 1,) unscaled closes used to compute pct returns
    """
    if len(features) <= window_size:
        raise ValueError("Not enough rows to build even one window.")
    windows = []
    for start in range(len(features) - window_size):
        windows.append(features[start : start + window_size])

    close_series = closes[window_size - 1 :]
    logger.info(
        "Built %d windows with window_size=%d and feature_size=%d",
        len(windows),
        window_size,
        features.shape[1],
    )
    return np.array(windows, dtype=np.float32), np.array(close_series, dtype=np.float32)


class TradingEnv:
    """Minimal trading environment with hold/long/short and P&L rewards on percent returns."""

    def __init__(
        self,
        windows: np.ndarray,
        close_series: np.ndarray,
        transaction_cost: float = 0.0002,
        reward_clip: Optional[float] = None,
        hold_bonus: float = 1e-4,
        position_penalty: float = 5e-5,
    ) -> None:
        """
        Args:
            windows: (steps, window_size, feature_size) prebuilt feature windows
            close_series: (steps + 1,) raw closes aligned with windows
            transaction_cost: flat cost applied when switching position
            reward_clip: clip reward to +/- this value (set None to disable)
            hold_bonus: small bonus when staying flat
            position_penalty: small per-step penalty when holding a position
        """
        if len(close_series) != len(windows) + 1:
            raise ValueError("close_series must be one element longer than windows")
        self.windows = windows
        self.close_series = close_series
        self.window_size = windows.shape[1]
        self.base_feature_size = windows.shape[2]
        self.action_size = 3  # 0: hold, 1: long, 2: short
        self.transaction_cost = transaction_cost
        self.reward_clip = reward_clip
        self.hold_bonus = hold_bonus
        self.position_penalty = position_penalty
        self.max_idx = len(windows) - 1
        self.cumulative_profit = 0.0  # True P&L sum
        self.num_trades = 0
        self.num_profitable_trades = 0
        self.reset()
        logger.info(
            "TradingEnv initialized: steps=%d, window_size=%d, feature_size=%d, transaction_cost=%.4f, reward_clip=%s, hold_bonus=%s, position_penalty=%s",
            len(windows),
            self.window_size,
            self.base_feature_size,
            transaction_cost,
            reward_clip,
            hold_bonus,
            position_penalty,
        )
    
    def get_metrics(self) -> dict:
        """Return current episode metrics."""
        win_rate = self.num_profitable_trades / max(1, self.num_trades)
        return {
            "cumulative_profit": self.cumulative_profit,
            "num_trades": self.num_trades,
            "num_profitable_trades": self.num_profitable_trades,
            "win_rate": win_rate,
        }

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.idx = 0
        self.position = 0  # -1 short, 0 flat, 1 long
        self.cumulative_profit = 0.0
        self.num_trades = 0
        self.num_profitable_trades = 0
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
        price_change_pct = (price_next - price_now) / price_now

        # Base P&L component
        pnl = self.position * price_change_pct

        # Apply transaction cost only when position changes (entry/flip)
        if self.position != prev_position:
            pnl -= self.transaction_cost
            self.num_trades += 1

        # Track profitability when closing or flipping
        if prev_position != 0 and self.position != prev_position:
            if pnl > 0:
                self.num_profitable_trades += 1

        # Reward shaping: incentivize holding flat when no strong signal
        shaping = 0.0
        if action == 0 and self.position == 0:
            shaping += self.hold_bonus

        # Penalize holding positions (opportunity cost)
        if self.position != 0:
            shaping -= self.position_penalty

        reward = pnl + shaping
        self.cumulative_profit += pnl
        
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.idx += 1
        done = self.idx >= self.max_idx
        next_state = (
            self._augment_state(self.windows[self.idx]) if not done else self._augment_state(self.windows[self.max_idx])
        )
        return next_state, reward, done


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer: List = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.frame = 1

    def add(self, state, action, reward, next_state, done) -> None:
        max_prio = self.priorities[: len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        if not self.buffer:
            raise ValueError("ReplayBuffer is empty")
        prios = self.priorities[: len(self.buffer)] ** self.alpha
        probs = prios / prios.sum()
        idxs = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*samples)

        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        weights = (len(self.buffer) * probs[idxs]) ** (-beta)
        weights /= weights.max()
        self.frame += 1

        # Convert to TensorFlow tensors for GPU efficiency
        return (
            tf.convert_to_tensor(np.stack(states), dtype=tf.float32),
            tf.convert_to_tensor(np.array(actions, dtype=np.int32), dtype=tf.int32),
            tf.convert_to_tensor(np.array(rewards, dtype=np.float32), dtype=tf.float32),
            tf.convert_to_tensor(np.stack(next_states), dtype=tf.float32),
            tf.convert_to_tensor(np.array(dones, dtype=np.float32), dtype=tf.float32),
            idxs,
            tf.convert_to_tensor(weights.astype(np.float32), dtype=tf.float32),
        )

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        self.priorities[indices] = np.abs(errors) + 1e-6

    def __len__(self) -> int:
        return len(self.buffer)


def create_q_model(window_size: int, feature_size: int, action_size: int) -> Model:
    inputs = Input(shape=(window_size, feature_size))

    conv_branch = Conv1D(96, kernel_size=3, activation="relu")(inputs)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Dropout(0.3)(conv_branch)
    conv_branch = Flatten()(conv_branch)

    lstm_branch = LSTM(96, return_sequences=True)(inputs)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Dropout(0.3)(lstm_branch)
    lstm_branch = Flatten()(lstm_branch)

    x = Concatenate()([conv_branch, lstm_branch])
    x = Dense(160, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(96, activation="relu")(x)

    # Dueling heads
    value = Dense(96, activation="relu")(x)
    value = Dense(1, activation=None)(value)

    advantage = Dense(96, activation="relu")(x)
    advantage = Dense(action_size, activation=None)(advantage)

    advantage_mean = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
    advantage_centered = Lambda(lambda a: a[0] - a[1])([advantage, advantage_mean])
    outputs = Lambda(lambda elems: elems[0] + elems[1])([value, advantage_centered])
    
    # Force float32 output for numerical stability with mixed precision
    outputs = Lambda(lambda x: tf.cast(x, tf.float32))(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=Huber(),
        jit_compile=True  # Enable XLA compilation for GPU
    )
    return model


@tf.function(reduce_retracing=True)
def train_step(model, target_model, s_batch, a_batch, r_batch, ns_batch, d_batch, gamma, weights):
    """GPU-optimized training step with prioritized experience replay."""
    with tf.GradientTape() as tape:
        # Current Q values
        current_q = model(s_batch, training=True)
        
        # Double DQN: use online network to select actions, target network to evaluate
        next_q_online = model(ns_batch, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
        
        next_q_target = target_model(ns_batch, training=False)
        next_indices = tf.stack([tf.range(tf.shape(next_actions)[0]), next_actions], axis=1)
        max_next_q = tf.gather_nd(next_q_target, next_indices)
        
        # Compute target
        target = r_batch + gamma * max_next_q * (1.0 - d_batch)
        target = tf.clip_by_value(target, -5.0, 5.0)
        
        # Get Q values for taken actions
        indices = tf.stack([tf.range(tf.shape(a_batch)[0]), a_batch], axis=1)
        current_q_taken = tf.gather_nd(current_q, indices)
        
        # Compute TD errors for priority updates
        td_errors = target - current_q_taken
        
        # Weighted loss for prioritized experience replay
        losses = Huber()(target, current_q_taken)
        weighted_loss = tf.reduce_mean(weights * losses)
    
    # Compute and apply gradients
    gradients = tape.gradient(weighted_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return weighted_loss, td_errors


def train_dqn(
    env: TradingEnv,
    model: Model,
    target_model: Model,
    episodes: int = 5,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay: float = 0.9995,  # Exponential decay rate
    warmup: int = 2000,
    target_update: int = 250,
    log_every: int = 500,
    max_steps_per_episode: int = 20000,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 100000,
    buffer_capacity: int = 50000,
) -> Tuple[List[float], List[dict]]:
    logger.info(
        "Starting DQN training: episodes=%d, batch_size=%d, warmup=%d, target_update=%d, epsilon_decay=%.4f, max_steps_per_episode=%d, per_alpha=%.2f, per_beta_start=%.2f, per_beta_frames=%d",
        episodes,
        batch_size,
        warmup,
        target_update,
        epsilon_decay,
        max_steps_per_episode,
        per_alpha,
        per_beta_start,
        per_beta_frames,
    )
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        alpha=per_alpha,
        beta_start=per_beta_start,
        beta_frames=per_beta_frames,
    )
    rewards_history: List[float] = []
    metrics_history: List[dict] = []
    step_count = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        loss_value = None
        steps_this_episode = 0
        action_counts = {0: 0, 1: 0, 2: 0}  # Track action distribution
        logger.info("Episode %d/%d started with epsilon=%.3f", ep + 1, episodes, epsilon)

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_size)
            else:
                q_values = model.predict(state[None, ...], verbose=0)[0]
                action = int(np.argmax(q_values))
            
            action_counts[action] += 1

            next_state, reward, done_env = env.step(action)
            step_limit_reached = steps_this_episode + 1 >= max_steps_per_episode
            done = done_env or step_limit_reached
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            step_count += 1
            steps_this_episode += 1
            epsilon = max(epsilon * epsilon_decay, epsilon_min)  # Exponential decay

            if len(buffer) == warmup:
                logger.info("Replay buffer warmup reached (%d transitions)", len(buffer))

            if len(buffer) >= warmup:
                (
                    s_batch,
                    a_batch,
                    r_batch,
                    ns_batch,
                    d_batch,
                    idxs,
                    weights,
                ) = buffer.sample(batch_size)

                # Use GPU-optimized training step
                loss_tensor, td_errors_tensor = train_step(
                    model, target_model,
                    s_batch, a_batch, r_batch, ns_batch, d_batch,
                    gamma, weights
                )
                loss_value = float(loss_tensor.numpy())
                td_errors = td_errors_tensor.numpy()
                
                # Update priorities in replay buffer
                buffer.update_priorities(idxs, td_errors)

                if step_count % target_update == 0:
                    target_model.set_weights(model.get_weights())
                    logger.info("Target network updated at step %d", step_count)

            if log_every and step_count % log_every == 0:
                env_metrics = env.get_metrics()
                logger.info(
                    "Step %d | ep=%d | ε=%.3f | reward=%.5f | profit=%.5f | trades=%d | win_rate=%.2f%% | buffer=%d | loss=%s",
                    step_count,
                    ep + 1,
                    epsilon,
                    ep_reward,
                    env_metrics["cumulative_profit"],
                    env_metrics["num_trades"],
                    env_metrics["win_rate"] * 100,
                    len(buffer),
                    "n/a" if loss_value is None else f"{loss_value:.6f}",
                )

        rewards_history.append(ep_reward)
        env_metrics = env.get_metrics()
        metrics_history.append(env_metrics)
        
        action_dist = {k: v / max(1, steps_this_episode) * 100 for k, v in action_counts.items()}
        logger.info(
            "Episode %d finished: reward=%.4f | profit=%.4f | trades=%d | win_rate=%.2f%% | ε=%.3f | actions: hold=%.1f%% long=%.1f%% short=%.1f%%",
            ep + 1,
            ep_reward,
            env_metrics["cumulative_profit"],
            env_metrics["num_trades"],
            env_metrics["win_rate"] * 100,
            epsilon,
            action_dist[0],
            action_dist[1],
            action_dist[2],
        )

    return rewards_history, metrics_history


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Starting Bitcoin price predictor training script")
    
    # Setup GPU/accelerator
    has_gpu = setup_gpu()
    logger.info(f"Training will run on: {'GPU' if has_gpu else 'CPU (or Metal on Apple Silicon)'}")
    
    data_dir = "data/klines"
    if not Path(data_dir).exists():
        raise SystemExit(f"Data directory {data_dir} not found")

    frame = load_klines(data_dir)
    features, closes = engineer_features(frame)

    window_size = 64
    windows, close_series = build_windows(features, closes, window_size)

    # Include position as an extra feature channel
    feature_size = windows.shape[2] + 1
    env = TradingEnv(
        windows,
        close_series,
        transaction_cost=0.0002,
        reward_clip=None,
        hold_bonus=1e-4,
        position_penalty=5e-5,
    )

    q_model = create_q_model(window_size, feature_size, env.action_size)
    target_q_model = create_q_model(window_size, feature_size, env.action_size)
    target_q_model.set_weights(q_model.get_weights())
    logger.info(
        "Models initialized; window_size=%d feature_size=%d action_size=%d",
        window_size,
        feature_size,
        env.action_size,
    )

    rewards, metrics = train_dqn(
        env,
        q_model,
        target_q_model,
        episodes=3,
        batch_size=64,
        warmup=2000,
        epsilon_decay=0.9997,  # Slower decay for better exploration
        log_every=500,
        max_steps_per_episode=15000,  # Shorter episodes initially
    )
    plot_rewards(rewards)
    
    # Print final metrics
    logger.info("Training completed. Final episode metrics:")
    final_metrics = metrics[-1]
    logger.info(
        "  Cumulative profit: %.4f | Total trades: %d | Profitable trades: %d | Win rate: %.2f%%",
        final_metrics["cumulative_profit"],
        final_metrics["num_trades"],
        final_metrics["num_profitable_trades"],
        final_metrics["win_rate"] * 100,
    )
