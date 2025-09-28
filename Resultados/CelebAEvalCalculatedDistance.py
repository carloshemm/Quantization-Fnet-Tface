import pandas as pd
from typing import Callable, Tuple

def compare_with_reference_ids(
    main_df: pd.DataFrame,
    reference_path: str,
    main_name_col: str = 'nome',
    main_match_col: str = 'nome_match',
    verbose: bool = True
) -> pd.DataFrame:
    

    ref_df = pd.read_csv(reference_path, sep=' ', header=None, names=['nome', 'id'])
    
    main_df = main_df.merge(
        ref_df,
        left_on=main_name_col,
        right_on='nome',
        how='left'
    ).rename(columns={'id': 'original_id'})
    
    main_df = main_df.merge(
        ref_df,
        left_on=main_match_col,
        right_on='nome',
        how='left'
    ).rename(columns={'id': 'matched_id'})
    
    main_df['id_match'] = main_df['original_id'] == main_df['matched_id']
    
    success_rate = main_df['id_match'].mean()
    total_rows = len(main_df)
    success_count = main_df['id_match'].sum()
    
    if verbose:
        print(f"\nComparison Results (using reference IDs):")
        print(f"Total rows: {total_rows:,}")
        print(f"Matching IDs: {success_count:,} ({success_rate:.2%})")
        

        non_matches = main_df[~main_df['id_match']]
   
    return main_df


if __name__ == "__main__":
    df = pd.read_csv("CelebaEmbeddingDistances.csv")
    
    path = "identity_CelebA.txt"
    
    result_df = compare_with_reference_ids(
    df,
    reference_path=path,
    main_name_col='nome',
    main_match_col='nome_match'
)