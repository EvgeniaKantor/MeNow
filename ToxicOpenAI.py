import openai
import pandas as pd
import os
import time
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
df = pd.read_excel('df (2).xlsx')
# Add an empty column 'Toxic_OpenAI'
df['Toxic_OpenAI'] = None

# Initialize the OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_toxicity(abstract):
    prompt_for_role_system = "You're a highly qualified biologist."
    prompt_for_role_user = (
        f"Determine if Withania Somnifera is toxic to humans based on the following abstract ("
        f"Respond with 'Toxic' or 'Nontoxic').\n\nAbstract:\n{abstract}"
    )

    # Create a chat completion request
    chat_gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_for_role_system},
            {"role": "user", "content": prompt_for_role_user}
        ]
    )

    # Extract and return the response content
    return chat_gpt_response['choices'][0]['message']['content'].strip()

# Rate limiting parameters
delay_between_requests = 1  # Delay in seconds between requests

# Construct toxicity dictionary from the DataFrame
toxicity_dict = {index: row['Abstract'] for index, row in df.iterrows()}

for toxicity_key, toxicity_value in toxicity_dict.items():
    try:
        if pd.isna(df.at[toxicity_key, 'Toxic_OpenAI']) or df.at[toxicity_key, 'Toxic_OpenAI'] == '':  # Check if cell is empty
            logging.info(f"Processing index {toxicity_key}...")

            response = generate_toxicity(toxicity_value)

            # Add the generated response to the DataFrame
            df.at[toxicity_key, 'Toxic_OpenAI'] = response
            logging.info(f"Generated toxicity information for index {toxicity_key}: {response}")

            time.sleep(delay_between_requests)  # Delay between requests

    except Exception as e:
        logging.error(f"Failed to generate toxicity information for index {toxicity_key}: {e}")

# Save the updated DataFrame to a new Excel file
df.to_excel('df_with_toxicity_info.xlsx', index=False)
