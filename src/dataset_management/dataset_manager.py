from pandas import DataFrame, read_csv
from pandas.errors import EmptyDataError, ParserError
from src.errors import ConfigError
import logging

_REQUIRED_COLUMNS = ("textid", "plain", "metaphor_tagged_text")
_TYPED_COLUMN = "metaphor_tagged_text_with_type"      # optional: only needed to pick examples by metaphor type

logger = logging.getLogger("uvicorn.error")

class DatasetManager:
    dataset_path: str
    _dataset_dfdataset_df: DataFrame

    def __init__(self, dataset_path: str) -> None:
        
        self.dataset_path = dataset_path
        try:
            dataset_df: DataFrame = read_csv(dataset_path)
        except FileNotFoundError as e:
            raise ConfigError(f"Dataset file not found: {dataset_path}") from e
        except (OSError, UnicodeDecodeError, EmptyDataError, ParserError) as e:
            raise ConfigError(f"Dataset file could not be read ({dataset_path}): {e}") from e
        
        missing_columns = set(_REQUIRED_COLUMNS) - set(dataset_df.columns)
        if missing_columns:
            raise ConfigError(f"Dataset is missing required columns: {sorted(missing_columns)}")
        if dataset_df.empty:
            raise ConfigError(f"Dataset {dataset_path} has no rows.")

        problems: list[str] = []
        duplicated_ids = dataset_df.loc[dataset_df["textid"].duplicated(), "textid"]
        if not duplicated_ids.empty:
            problems.append(f"textid must be unique; repeated: {duplicated_ids.unique()[:5].tolist()}")
        for column in _REQUIRED_COLUMNS:
            blank = dataset_df[column].isna() | (dataset_df[column].astype(str).str.strip() == "")
            if blank.any():
                rows = (dataset_df.index[blank] + 2)[:5].tolist()          # +2: header line, and CSV rows start at 1
                problems.append(f"'{column}' is empty or missing in {int(blank.sum())} row(s), first at CSV line(s) {rows}")
        if problems:
            raise ConfigError(f"Dataset {dataset_path} is invalid:\n- " + "\n- ".join(problems))

        self._dataset_df = dataset_df

    @property
    def has_types(self) -> bool:
        return _TYPED_COLUMN in self._dataset_df.columns

    def get_dataset(self) -> DataFrame:
        """A copy, so no node can change the shared data by accident."""
        return self._dataset_df.copy()