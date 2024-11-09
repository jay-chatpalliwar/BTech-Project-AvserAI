import pandas as pd

# Load the JSON file
json_file_path = 'demo2.json'
data = pd.read_json(json_file_path)

# Convert to CSV
csv_file_path = 'data.csv'
data.to_csv(csv_file_path, index=False)

print(f"Converted {json_file_path} to {csv_file_path}")
