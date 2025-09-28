import pandas as pd
from typing import Callable, Tuple

def splitgetzero(nome):
    return nome.split("/")[0]

def compare_columns(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    comparison_func: Callable = lambda x, y: x==y,
    result_col: str = 'is_match',
    verbose: bool = True
) -> Tuple[pd.DataFrame, float]:

    # Perform the comparison
    df[result_col] = df.apply(
        lambda row: comparison_func(row[col1], row[col2]), 
        axis=1
    )
    
    # Calculate success metrics
    success_rate = df[result_col].mean()
    total_rows = len(df)
    success_count = df[result_col].sum()
    
    if verbose:
        print("\nComparison Results:")
        print(f"DataFrame size: {total_rows} rows")
        print(f"Matching rows: {success_count}")
        print(f"Success rate: {success_rate:.2%}\n")
        print("Sample of non-matching rows (if any):")
        if not df[result_col].all():
            print(df[~df[result_col]].head())
    
    return df, success_rate


if __name__ == "__main__":
    df = pd.read_csv("/media/bunto22/6894-E9551/Carlos/Quantization-Fnet-Tface/Resultados/output.csv")
    
    dfcomp = compare_columns(
    df,
    'nome',
    'nome_match',
    comparison_func=lambda x, y: splitgetzero(x) == splitgetzero(y),
)