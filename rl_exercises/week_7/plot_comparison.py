import pandas as pd
import matplotlib.pyplot as plt

# Paths to your CSV files (local to week_7 folder)
rnd_csv = 'training_data_rnd.csv'
baseline_csv = 'training_data_baseline.csv'

# Load data
rnd_data = pd.read_csv(rnd_csv)
baseline_data = pd.read_csv(baseline_csv)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(rnd_data['steps'], rnd_data['rewards'], label='RND DQN')
plt.plot(baseline_data['steps'], baseline_data['rewards'], label='Baseline DQN (Epsilon-Greedy)')
plt.xlabel('Steps')
plt.ylabel('Episode Reward')
plt.title('RND DQN vs Baseline DQN on LunarLander-v3')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('comparison_plot.png')
plt.show()
