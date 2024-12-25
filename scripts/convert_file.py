import pandas as pd

# Define the paths
input_file_path = './data/MachineLearningRating_v3.txt'  # Adjust the input file name as needed
output_file_path = './data/MLRatings.csv'

# Read the text file and write to CSV
df = pd.read_csv(input_file_path, delimiter='|')
df.to_csv(output_file_path, index=False)

print(f"File converted and saved to {output_file_path}")
