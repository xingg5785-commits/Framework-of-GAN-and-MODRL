import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gymnasium as gym
from scipy.stats import entropy
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ====== Import the dataset ======
df = pd.read_csv(r"C:\Users\郭兴\OneDrive\桌面\Level_Metadta_Comment.csv")

# ===== Handling missing value ======
print(df.isnull().sum())

# Remove the missing values
df = df.dropna(subset=['text'])
df['description'] = df['description'].fillna('')

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

categorical_cols = df.select_dtypes(include=['object']).columns
categorical_imputer = SimpleImputer(strategy='most_frequent', fill_value='missing')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

print(df.isnull().sum())

# ====== Encoding categorical data ======
data_column_category = df.select_dtypes (exclude=[np.number]).columns
print(data_column_category)
print(df[data_column_category].head())

label_encoder = LabelEncoder()
for i in data_column_category:
  df[i] = label_encoder.fit_transform (df[i])

print("Label Encoder Data:")
print(df.head())

# ====== Feature scaling ======
numeric_cols = df.select_dtypes(include=[np.number]).columns

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("Scaled Data:")
print(df.head())

# ======= Feature Importance ======
features = [
    "gamestyle", "theme", "difficulty", "game_version", "upload_time",
    "clear_condition", "clear_condition_magnitude", "timer", "autoscroll_speed"
]

targets = [
    "world_record", "clears", "attempts", "clear_rate", "plays", "likes", "boos",
    "weekly_likes", "weekly_plays", "unique_players_and_versus"
]

# ====== Random Forest for feature importance ======
importance_dict = {f:[] for f in features}
for target in targets:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df[target])
    importance = model.feature_importances_
    for f,imp in zip(features, importance):
        importance_dict[f].append(imp)

# ====== Average importance ======
avg_importance = {f: np.mean(importance_dict[f]) for f in features}
sorted_feat = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

# ====== Visualization ======
print("\nAverage feature importances:")
for f, s in sorted_feat:
    print(f"{f:<30} {s:.4f}")

plt.figure(figsize=(10,6))
plt.bar([f[0] for f in sorted_feat], [f[1] for f in sorted_feat])
plt.title("Average Feature Importance for GAN Condition Inputs")
plt.xticks(rotation=45)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# ====== Random Forest for target importance ======
reward_target = "clear_rate"

# Train a Random Forest to evaluate how other targets relate to this one
importance_dict = {t: [] for t in targets if t != reward_target}

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(df[[t for t in targets if t != reward_target]], df[reward_target])
importances = rf.feature_importances_

for t, imp in zip([t for t in targets if t != reward_target], importances):
    importance_dict[t].append(imp)

avg_importance = {t: np.mean(importance_dict[t]) for t in importance_dict}
sorted_targets = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

print("\nAverage importance of original targets for predicting reward (clear_rate):")
for t, imp in sorted_targets:
    print(f"{t:<30} {imp:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(
    [t for t, _ in sorted_targets],
    [imp for _, imp in sorted_targets]
)
plt.title("Feature Importance of Target Variables for Proxy Reward (clear_rate)")
plt.xlabel("Target Variables")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ====== cGAN Training ======
# Configuration
NEW_SAMPLE_SIZE = 2000
BATCH_SIZE  = 64
LATENT_DIM  = 100
EPOCHS      = 5000
SAVE_EVERY  = 1000

# Select condition columns
cond_cols = ["timer", "upload_time", "gamestyle", "difficulty"]
cond_scaler = MinMaxScaler(feature_range=(0, 1))
cond_data = scaler.fit_transform(df[cond_cols])
real_data = cond_data.copy()
data_dim  = real_data.shape[1]
cond_dim  = cond_data.shape[1]

# ====== Define model ======
def build_generator(latent_dim, cond_dim, output_dim):
    noise_in  = layers.Input(shape=(latent_dim,))
    cond_in   = layers.Input(shape=(cond_dim,))
    x = layers.Concatenate()([noise_in, cond_in])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(output_dim, activation="sigmoid")(x)
    return Model([noise_in, cond_in], x, name="Generator")

def build_discriminator(data_dim, cond_dim):
    sample_in = layers.Input(shape=(data_dim,))
    cond_in   = layers.Input(shape=(cond_dim,))
    x = layers.Concatenate()([sample_in, cond_in])
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    return Model([sample_in, cond_in], x, name="Discriminator")

# Build models
generator = build_generator(LATENT_DIM, cond_dim, data_dim)
discriminator = build_discriminator(data_dim, cond_dim)

# Define optimizer
opt_d = Adam(1e-4, beta_1=0.5)

# compile discriminator (before freezing)
discriminator.compile(optimizer=opt_d, loss="binary_crossentropy")

# Freeze the discriminator parameters and construct the cGAN
discriminator.trainable = False
noise_in  = layers.Input(shape=(LATENT_DIM,))
cond_in   = layers.Input(shape=(cond_dim,))
gen_out   = generator([noise_in, cond_in])
validity  = discriminator([gen_out, cond_in])
gan_model = Model([noise_in, cond_in], validity ,name="cGAN")

# Compile gan model
opt_gan = Adam(1e-4, beta_1=0.5)
gan_model.compile(optimizer=opt_gan, loss="binary_crossentropy")

# Labels
real_label = np.random.uniform(0.9, 1.0, size=(BATCH_SIZE, 1))
fake_label = np.random.uniform(0.0, 0.1, size=(BATCH_SIZE, 1))
real_label = np.ones((BATCH_SIZE, 1))
fake_label = np.zeros((BATCH_SIZE, 1))

# ====== Train Loop ======
loss_log = []

# Training discriminator
for epoch in range(1, EPOCHS + 1):
    tf.config.run_functions_eagerly(True)  # Enable eager execution to allow tensor.numpy()
    discriminator.trainable = True
    idx = np.random.randint(0, real_data.shape[0], BATCH_SIZE)
    real_samples = real_data[idx]
    cond_samples = cond_data[idx]

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_samples = generator.predict([noise, cond_samples], verbose=0)

    # Upgrade discriminator
    d_loss_real = discriminator.train_on_batch([real_samples, cond_samples], real_label)
    d_loss_fake = discriminator.train_on_batch([fake_samples, cond_samples], fake_label)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Training generator
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = gan_model.train_on_batch([noise, cond_samples], real_label)

    loss_log.append([epoch, d_loss, g_loss])

    if epoch % SAVE_EVERY == 0 or epoch == 1:
        print(f"Epoch {epoch}/{EPOCHS}  D loss: {d_loss:.4f}  G loss: {g_loss:.4f}")

# Visualization of loss trend
epochs, d_losses, g_losses = zip(*loss_log)
plt.plot(epochs, d_losses, label='Discriminator Loss')
plt.plot(epochs, g_losses, label='Generator Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curves")
plt.grid(True)
plt.show()

# Use the trained generator to generate new samples
noise_new = np.random.normal(0, 1, (NEW_SAMPLE_SIZE, LATENT_DIM))
cond_example = np.tile(cond_data.mean(axis=0), (NEW_SAMPLE_SIZE, 1))
gen_levels = generator.predict([noise_new, cond_example], verbose=0)
gen_levels_orig = scaler.inverse_transform(gen_levels)

# Save to csv
#df_new = pd.DataFrame(gen_levels_orig, columns=cond_cols).round(2)
#df_new.to_csv("synthetic_levels.csv", index=False)
#print("\nSaved synthetic levels to 'synthetic_levels.csv'")

# ====== Load synthetic levels and define difficulty scheduler ======
WINDOW_SIZE, r_min, r_max, alpha = 20, 0.4, 0.6, 0.1
d_t, perf_window = 0.5, []
targets = ["world_record", "clears"]


orig_df = pd.read_csv(r"C:\Users\郭兴\OneDrive\桌面\Level_Metadta_Comment.csv")
orig_df = orig_df.dropna(subset=targets)[targets].astype(np.float32).reset_index(drop=True)

scaler = MinMaxScaler()
obs_data = scaler.fit_transform(orig_df.values)

# Set data_dim based on available numeric data
data_dim = obs_data.shape[1]

def update_difficulty(perf, d):
    if len(perf) >= WINDOW_SIZE:
        avg = sum(perf) / len(perf)
        if avg > r_max:
            d = min(1.0, d + alpha * (avg - r_max))
        elif avg < r_min:
            d = max(0.0, d - alpha * (r_min - avg))
        perf.clear()
    return d

class LevelEnv(gym.Env):
    def __init__(self, obs_data, info_df):
        super(LevelEnv, self).__init__()
        self.obs_data = obs_data
        self.info_df = info_df
        self.idx = 0
        self.observation_space = spaces.Box(0.0, 1.0, shape=(data_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(data_dim)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        return self.obs_data[self.idx], {}

    def step(self, action):
        global d_t, perf_window
        self.idx = (self.idx + 1) % len(self.obs_data)
        obs = self.obs_data[self.idx]

        entry = self.info_df.iloc[self.idx]

        attempts = entry.get("attempts", entry["clears"])
        clear_rate = entry["clears"] / max(1, attempts)
        world_record_sim = min(entry["world_record"] / 1000, 1.0)
        clears_sim = min(entry["clears"], 1000)

        likes = entry.get("likes", 0)
        boos = entry.get("boos", 0)
        likes_ratio = likes / max(1, likes + boos)

        reward = float(
            0.3 * clear_rate +
            0.2 * (clears_sim / 1000) +
            0.2 * (1.0 - world_record_sim) +
            0.15 * (likes / 1000) +
            0.15 * likes_ratio
        )

        perf_window.append(clear_rate)
        d_t = update_difficulty(perf_window, d_t)
        info = {"clear_rate": clear_rate, "difficulty": d_t}
        return obs, reward, False, False, info

# ====== Train PPO Model on Environment ======
env = LevelEnv(obs_data, orig_df)
check_env(env)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# ====== Evaluate and Record Results ======
synth = pd.read_csv(r"C:\\Users\\郭兴\\OneDrive\\桌面\\synthetic_levels.csv")
orig_targets = orig_df.iloc[:len(synth)].reset_index(drop=True)

records = []
obs, _ = env.reset()
for i in range(len(synth)):
    action, _ = model.predict(obs)
    obs, reward, _, _, info = env.step(action)
    entry = {f"feature_{j}": float(v) for j, v in enumerate(obs)}
    entry.update({
        "reward": float(reward),
        "clear_rate_modrl": float(info["clear_rate"]),
        "difficulty": float(info["difficulty"])
    })
    for t in targets:
        entry[t] = float(orig_targets.at[i, t])
    records.append(entry)

out = pd.DataFrame(records)
corr = out[["reward"] + targets].corr()["reward"].drop("reward")
print("\nCorrelation between reward and targets:\n", corr)

#out.to_csv("generated_levels_worldrecord_clears.csv", index=False)
#print("\nSaved to 'generated_levels_worldrecord_clears.csv'")

# === Structural Diversity Analysis ===
def spatial_entropy(matrix, bins=10):
    hist, _ = np.histogram(matrix.flatten(), bins=bins, density=True)
    return entropy(hist)

orig_levels = pd.read_csv(r"C:\Users\郭兴\OneDrive\桌面\Level_Metadta_Comment.csv")
orig_numeric = orig_levels.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy="mean")
orig_numeric = imputer.fit_transform(orig_numeric)
orig_entropy = [spatial_entropy(row) for row in orig_numeric]
mean_orig_ent = np.mean(orig_entropy)
print(f"\nOriginal dataset entropy: {mean_orig_ent:.3f}")

ppo_levels = pd.read_csv(r"C:\Users\郭兴\OneDrive\桌面\Synethetic Levels_Proxy_Reward.csv")
ppo_values = ppo_levels.select_dtypes(include=[np.number]).values
ppo_entropy = [spatial_entropy(row) for row in ppo_values]
mean_ppo_ent = np.mean(ppo_entropy)
print(f"\nPPO-optimized levels entropy: {mean_ppo_ent:.3f}")

gen_levels = pd.read_csv(r"C:\Users\郭兴\OneDrive\桌面\synthetic_levels.csv")
gen_values = gen_levels.select_dtypes(include=[np.number]).values
gen_entropy = [spatial_entropy(row) for row in gen_values]
mean_gen_ent = np.mean(gen_entropy)
print(f"\nAverage spatial entropy of generated levels {mean_gen_ent:.3f}")

labels = ['Original', 'PPO-Optimized', 'Generated']
values = [mean_orig_ent, mean_ppo_ent, mean_gen_ent]
colors = ['gray', 'lightgreen', 'skyblue']

# Visualization
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, values, color=colors)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.3f}",
             ha='center', va='bottom', fontsize=10)

plt.title('Comparison of Spatial Entropy')
plt.ylabel('Average Entropy')
plt.ylim(0, max(values)*1.2)
plt.grid(axis='y')
plt.tight_layout()
plt.show()