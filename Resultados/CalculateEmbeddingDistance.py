import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ast

def preprocess_dataframe(df):

    df_processed = df.copy()
    
    df_processed['embedding'] = df_processed['embedding'].apply(ast.literal_eval)
    
    return df_processed


def find_most_similar(row_index, row_data, batch_data, batch_indices):

    distances = cdist([row_data], batch_data, metric='cosine')

    if row_index in batch_indices:
        distances[0, batch_indices.index(row_index)] = np.inf
        
    return distances


def process_batch(row_index, dataset, batch_size=1000):

    num_rows = len(dataset)
    distance_results = np.zeros(num_rows)
    row_index = row_index[0]
    row_data = dataset[row_index]

    for batch_start in range(0, num_rows, batch_size):
        batch_end = min(batch_start + batch_size, num_rows)
        batch_data = dataset[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        result = find_most_similar(row_index, row_data, batch_data, batch_indices)
        distance_results[batch_start:batch_end] = result
        

    most_similar_index = np.argmin(distance_results)
    distance = distance_results[most_similar_index]


    
    return {row_index:[row_index, most_similar_index, distance]}


def find_all_most_similar(df, batch_size=1000, max_workers=6):

    dataset = np.array(df['embedding'].tolist()) 
    num_rows = len(dataset)
    results = {}

    batches = [range(i, min(i + 1, num_rows)) for i in range(0, num_rows, 1)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch, dataset, batch_size) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(batches), desc="Processing batches"):
            results.update(future.result())

    result_df = pd.DataFrame([values for values in results.values()], columns=['row_index', 'most_similar_index', 'similarity']).sort_values(by='row_index')
    
    
    matches_df = df.iloc[result_df['most_similar_index']]
    
    df['most_similar_index'] = result_df['most_similar_index'].to_numpy()
    df['distance'] = result_df['similarity'].to_numpy()
    df['nome_match'] = matches_df['nome'].to_numpy()
    
    df = df.drop('embedding', axis=1)
    
    return df

csv_file_path = '/media/bunto22/6894-E9551/Carlos/Quantization-Fnet-Tface/Transface/output.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)  # Assuming the first column contains row names

preprocessed_df = preprocess_dataframe(df)

result_df = find_all_most_similar(preprocessed_df, batch_size=1000)

output_csv_path = f'output.csv'
result_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")