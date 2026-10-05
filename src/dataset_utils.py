from pandas import DataFrame, read_csv

def load_dataset(filepath: str ='corpus/metaphor_dataset.csv') -> DataFrame:
    """
    Load the metaphor dataset.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame with metaphor data
    """
    df = read_csv(filepath)
    print(f'Dataset loaded successfully!')
    print(f'Shape: {df.shape}')
    return df

def dataset_details(dataset: DataFrame):
    print('='*60)
    print('DATASET FACTS')
    print('='*60)
    print(f'Total number of entries: {len(dataset)}')
    print(f'Number of columns: {len(dataset.columns)}')
    print(f'\nColumn names and types:')
    print(dataset.dtypes)
    print(f'\nMissing values:')
    print(dataset.isnull().sum())
    print(f'\nDataset shape: {dataset.shape}')
    print(f'\nFirst few entries:')
    print(dataset.head(2))

def load_dataset_for_rag(dataset_path: str ='corpus/metaphor_dataset.csv',
                         prompts_path: str = 'resources/prompts.csv',
                         row_limit: int | None = None) -> tuple[DataFrame, DataFrame]:
    dataset_df = read_csv(dataset_path)
    prompts_df = read_csv(prompts_path)

    if row_limit is not None:
        dataset_df = dataset_df.head(row_limit).copy()

    required_dataset_cols = {"textid", "plain", "metaphor_tagged_text"}
    missing_dataset_cols = required_dataset_cols - set(dataset_df.columns)
    if missing_dataset_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_dataset_cols}")

    required_prompt_cols = {"pid", "name", "cid", "role", "content"}
    missing_prompt_cols = required_prompt_cols - set(prompts_df.columns)
    if missing_prompt_cols:
        raise ValueError(f"Prompt file is missing required columns: {missing_prompt_cols}")

    print(f"Dataset rows loaded:      {len(dataset_df)}")
    print(f"Prompt rows loaded:       {len(prompts_df)}")
    print(f"Unique prompt strategies: {prompts_df['pid'].nunique()}")

    return dataset_df, prompts_df

def split_dataset_for_rag(dataset_df: DataFrame,
                          top_k: int = 3,
                          train_fraction: float = 0.8,
                          random_seed: int = 42) -> tuple[DataFrame, DataFrame]:
    train_df = dataset_df.sample(frac=train_fraction, random_state=random_seed)
    test_df  = dataset_df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    if len(train_df) < top_k:
        raise ValueError(
            f"Retrieval pool has only {len(train_df)} examples but TOP_K={top_k}. "
            "Increase TRAIN_FRACTION or reduce TOP_K."
        )

    print(f"Retrieval pool size: {len(train_df)}")
    print(f"Test set size:       {len(test_df)}")
    return train_df, test_df
