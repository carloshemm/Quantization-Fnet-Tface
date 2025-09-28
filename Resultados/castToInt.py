import pandas as pd
import ast  # for safely evaluating strings as lists
import numpy as np

# Read the CSV file
df = pd.read_csv('/media/bunto22/6894-E9551/Carlos/Quantization-Fnet-Tface/Transface/output.csv')

# Function to convert embedding to int8
def convert_to_int8(embedding_str):
    # Convert string representation of list to actual list
    embedding_list = ast.literal_eval(embedding_str)
    # Multiply each value by 127 and convert to int8
    return np.array([int(round(x * 127)) for x in embedding_list], dtype=np.int8).tolist()

# Apply the conversion to the embedding column
df['embedding'] = df['embedding'].apply(convert_to_int8)

# Save the modified dataframe back to CSV
df.to_csv('output-int.csv', index=False)