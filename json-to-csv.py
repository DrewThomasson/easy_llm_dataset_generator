import pandas as pd
import json

# Load the JSON data from file
with open('lewd.json', 'r') as file:  # Replace 'path_to_your_file.json' with your actual JSON file path
    data = json.load(file)

# Create a DataFrame
df = pd.DataFrame(data)
df = df[['prompt', 'chosen']].rename(columns={'prompt': 'Prompt', 'chosen': 'Response'})

# Save the DataFrame to a CSV file
df.to_csv('lewd.csv', index=False)  # This will create 'output_file.csv' with the desired columns

