import pandas as pd
from tqdm.auto import tqdm
import ollama

# Load your dataset
df = pd.read_csv('batman.csv')

# Define your system prompt
system_prompt = "You are the batman and you will always respond to everything in a dark and gloomy tone, always give a short and simple response"

# Function to get response from Ollama
def get_ollama_response(prompt):
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': f'{system_prompt}  {prompt}',
        },
    ])
    return response['message']['content']

# Iterate over DataFrame rows and update the 'Response' column with responses from Ollama
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    df.at[index, 'Response'] = get_ollama_response(row['Prompt'])

# Save the updated DataFrame back to a CSV file
df.to_csv('updated_batman.csv', index=False)

print("Dataset processing complete. Updated dataset saved as 'updated_batman.csv'.")
