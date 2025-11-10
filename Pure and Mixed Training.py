import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind

# ===== Define PureSimEnv =====
class PureSimEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.idx = 0
        n_features = data.shape[1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_features)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        return self.data[self.idx], {}

    def step(self, action):
        obs = self.data[self.idx]
        reward = float(obs[action])
        self.idx += 1
        done = self.idx >= len(self.data)
        next_obs = self.data[self.idx] if not done else np.zeros_like(obs)
        return next_obs, reward, done, False, {}

# ===== Loading synthetic dataset for pure training =====
csv_path = r"C:\Users\郭兴\OneDrive\桌面\Synethetic Levels_Proxy_Reward.csv"
df = pd.read_csv(csv_path)
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values)

# ===== Pure training =====
def make_env():
    return PureSimEnv(data)

os.makedirs("pure_logs", exist_ok=True)
base_env = DummyVecEnv([make_env])
vec_env = VecMonitor(base_env, filename=os.path.join("pure_logs", "monitor.csv"))
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_pure_sim")

# ===== Define MixedSimEnv =====
class MixedSimEnv(gym.Env):
    def __init__(self, data, episode_length=2000):
        super().__init__()
        self.data = data
        self.episode_length = episode_length
        self.step_count = 0
        n_features = data.shape[1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_features)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.indices = np.random.choice(len(self.data), size=self.episode_length, replace=True)
        self.step_count = 0
        obs = self.data[self.indices[self.step_count]]
        return obs, {}

    def step(self, action):
        idx = self.indices[self.step_count]
        obs_current = self.data[idx]
        reward = float(obs_current[action])
        self.step_count += 1
        done = self.step_count >= self.episode_length
        next_obs = self.data[self.indices[self.step_count]] if not done else np.zeros_like(obs_current)
        return next_obs, reward, done, False, {}

# ===== Mixed training =====
orig_path = r"C:\Users\郭兴\OneDrive\桌面\Level_Metadta_Comment.csv"
synth_path = r"C:\Users\郭兴\OneDrive\桌面\Synethetic Levels_Proxy_Reward.csv"
orig_df = pd.read_csv(orig_path)
synth_df = pd.read_csv(synth_path)
combined_df = pd.concat([orig_df, synth_df], ignore_index=True)
numeric_df = combined_df.select_dtypes(include=[np.number]).fillna(0)
combined_data = MinMaxScaler().fit_transform(numeric_df.values)

def make_mixed_env():
    return MixedSimEnv(combined_data)

os.makedirs("mixed_logs", exist_ok=True)
base_env = DummyVecEnv([make_mixed_env])
vec_env = VecMonitor(base_env, filename=os.path.join("mixed_logs", "monitor.csv"))
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_mixed_sim")

# ===== Plotting and Evaluation =====
pure_df = pd.read_csv("pure_logs/monitor.csv", skiprows=1)
mixed_df = pd.read_csv("mixed_logs/monitor.csv", skiprows=1)

pure_timesteps = pure_df["l"].cumsum()
mixed_timesteps = mixed_df["l"].cumsum()

pure_rewards_smooth = gaussian_filter1d(pure_df["r"], sigma=2)
mixed_rewards_smooth = gaussian_filter1d(mixed_df["r"], sigma=2)

plt.figure(figsize=(10, 6))
plt.plot(pure_timesteps, pure_rewards_smooth, label="Pure Simulation (Smoothed)", linewidth=2)
plt.plot(mixed_timesteps, mixed_rewards_smooth, label="Mixed Training (Smoothed)", linewidth=2)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Smoothed Learning Curves: Pure vs. Mixed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Statistical Evaluation =====
def print_stats(df, label):
    mean_reward = df["r"].mean()
    std_reward = df["r"].std()
    max_reward = df["r"].max()
    min_reward = df["r"].min()
    last_10_avg = df["r"].tail(10).mean()
    last_10_std = df["r"].tail(10).std()

    print(f"\n=== {label} ===")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Last 10 episodes - Avg: {last_10_avg:.2f}, Std: {last_10_std:.2f}")

print_stats(pure_df, "Pure Simulation")
print_stats(mixed_df, "Mixed Training")

pure_rewards = pure_df["r"]
mixed_rewards = mixed_df["r"]
boxplot_df = pd.DataFrame({
    "reward": pd.concat([pure_rewards, mixed_rewards], ignore_index=True),
    "Type": ["Pure"] * len(pure_rewards) + ["Mixed"] * len(mixed_rewards)
})

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=boxplot_df,
    x="Type",
    y="reward",
    hue="Type",
    palette={"Pure": "blue", "Mixed": "green"},
    legend=False
)
plt.title("Reward Distribution: Pure vs. Mixed Training")
plt.ylabel("Episode Reward")
plt.grid(True)
plt.tight_layout()
plt.show()
print("==================================")

# === Convergence Steps speed between pure and hybrid simulation training ===
def evaluate_model_steps(env, model, threshold, n_eval_envs=10):
    test_rewards = []
    for _ in range(n_eval_envs):
        obs, done = env.reset(), False
        total = 0.0
        while not (done if isinstance(done, bool) else all(done)):
            a, _ = model.predict(obs)
            res = env.step(a)
            if len(res) == 5:
                obs, r, done, trunc, _ = res
            else:
                obs, r, done, _ = res
            total += float(r[0] if isinstance(r, (list, np.ndarray)) else r)
        test_rewards.append(total)
    return np.mean(test_rewards) >= threshold

def run_trials(env_fn, threshold, trials=5, step_increment=1000, max_steps=10000):
    results = []
    for trial in range(trials):
        env = env_fn()
        model = PPO("MlpPolicy", env, seed=trial, policy_kwargs=dict(net_arch=[256,256]), verbose=0)
        total_steps = 0
        start_time = time.time()
        conv = max_steps

        while total_steps < max_steps:
            model.learn(total_timesteps=step_increment, reset_num_timesteps=False)
            total_steps += step_increment

            if evaluate_model_steps(env, model, threshold):
                conv = total_steps
                break
        elapsed = time.time() - start_time
        results.append((conv, elapsed))
    return results

threshold = 0.3 * max(mixed_df["r"].max(), pure_df["r"].max())

def make_pure_env():
    return DummyVecEnv([lambda: PureSimEnv(data)])

def make_mixed_env():
    return DummyVecEnv([lambda: MixedSimEnv(combined_data, episode_length=3000)])

pure_results = run_trials(make_pure_env, threshold=threshold,
                            trials=5, step_increment=1000, max_steps=10000)
hybrid_results = run_trials(make_mixed_env, threshold=threshold,
                            trials=5, step_increment=1000, max_steps=10000)

pure_steps, pure_times = zip(*pure_results)
hybrid_steps, hybrid_times = zip(*hybrid_results)

for i, (p_step, p_time) in enumerate(pure_results):
    print(f"Pure Trial {i+1}: Pure_steps = {p_step}, Time = {p_time:.2f}s")

for i, (h_step, h_time) in enumerate(hybrid_results):
    print(f"Hybrid Trial {i+1}: Hybrid_steps = {h_step}, Time = {h_time:.2f}s")

x = np.arange(1, len(pure_steps) + 1)
bar_width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Figure1: Converge step
ax1.bar(x - bar_width/2, pure_steps, width=bar_width, label='Pure Steps', color='skyblue')
ax1.bar(x + bar_width/2, hybrid_steps, width=bar_width, label='Hybrid Steps', color='salmon')
ax1.set_xlabel('Trial')
ax1.set_ylabel('Steps to Converge')
ax1.set_title('Convergence Steps per Trial')
ax1.set_xticks(x)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Figure2: Training time
ax2.bar(x - bar_width/2, pure_times, width=bar_width, label='Pure Time (s)', color='deepskyblue')
ax2.bar(x + bar_width/2, hybrid_times, width=bar_width, label='Hybrid Time (s)', color='lightcoral')
ax2.set_xlabel('Trial')
ax2.set_ylabel('Training Time (s)')
ax2.set_title('Training Time per Trial')
ax2.set_xticks(x)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()