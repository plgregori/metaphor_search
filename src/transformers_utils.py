from pandas import DataFrame

def prepare_finetuning_data_transformers(dataset: DataFrame, train_ratio: float = 0.8, seed: int = 1):
    train_df = dataset.sample(frac=train_ratio, random_state=seed)
    test_df = dataset.drop(index=train_df.index)
    return train_df, test_df