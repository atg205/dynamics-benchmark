import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the CSV file
df = pd.read_csv('best_results_pruned_1.csv')

# Filter out instances with 'embedded' in the name
df = df[~df['instance'].str.contains('embedded')]

# Drop rows where num_steps is 1000000
df = df[df['num_steps'] < 1_000_000]

# Extract the two numbers from the instance names
pattern = r'precision_(\d)_timepoints_(\d)\.coo'
df[['precision', 'timepoints']] = df['instance'].str.extract(pattern).astype(int)

# Create a 4x3 grid of subplots
fig, axes = plt.subplots(3,4, figsize=(20, 16))
fig.suptitle('Gap vs num_rep for different num_steps by instance type', y=1.02, fontsize=14)

# Iterate through each combination of precision and timepoints
for (precision, timepoints, nvar), group in df.groupby(['precision', 'timepoints', 'num_var']):
    # Determine the subplot position
    row = precision - 2  # precisions start at 2
    col = timepoints - 3  # timepoints start at 3
    
    # Create the plot for this instance type
    ax = axes[row, col]
    sns.lineplot(data=group, x='num_rep', y='gap', hue='num_steps', palette='Set1',
                 marker='o', ax=ax)
    
    ax.set_title(f'precision={precision}, timepoints={timepoints}, num_var={nvar}')
    ax.set_xlabel('NUM_REPS')
    ax.set_ylabel('Gap (%)')
    ax.set_xscale('log')
    ax.legend(title='NUM_STEPS')
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig("results.pdf", bbox_inches='tight')