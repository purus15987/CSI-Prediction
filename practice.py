import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the CSV file where NMSE results are saved
csv_filename = 'performance_metrics.csv'  # Make sure this matches the saved CSV file path

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_filename)

# Set up the seaborn style for better aesthetics
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=df, x='Speed (kmph)', y='RMSE', hue='Model', ci=None)

# Set the labels and title
ax.set_xlabel('Speed (kmph)', fontsize=14)
ax.set_ylabel('RMSE', fontsize=14)
ax.set_title(f'Comparison of RMSE for Models for Models at Different Speeds and Encoded Dims', fontsize=16)

# Show the plot
plt.tight_layout()
plt.savefig(f'RMSE_comparison.png', dpi=300)  # Save the plot as a PNG image
plt.show()