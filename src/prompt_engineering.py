from pandas import DataFrame, read_csv
from copy import deepcopy


def dataset_preparation(dataset_path: str, prompts_path: str, row_limit: int | None = None) -> tuple[DataFrame, DataFrame]:
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

    print(f"Dataset rows loaded: {len(dataset_df)}")
    print(f"Prompt rows loaded: {len(prompts_df)}")
    print(f"Unique prompt strategies: {prompts_df['pid'].nunique()}")

    return dataset_df, prompts_df

def load_prompt_strategies(prompt_table: DataFrame):
    prompt_messages_by_pid: dict[int, list[dict[str, str]]] = {}
    prompt_names_by_pid: dict[int, str] = {}

    for pid, group in prompt_table.sort_values(["pid", "cid"]).groupby("pid", sort=False):
        if not isinstance(pid, int):
            raise ValueError(f"Expected 'pid' to be int, got {type(pid).__name__} ({pid!r}). Check prompts.csv.")

        group = group.sort_values("cid").reset_index(drop=True)

        messages: list[dict[str, str]] = []
        for _, row in group.iterrows():
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"],
                }
            )
        prompt_messages_by_pid[pid] = messages
        prompt_name = group.loc[0, "name"]
        if not isinstance(prompt_name, str):
            raise ValueError(f"Expected 'name' to be str for pid={pid}, got {type(prompt_name).__name__} ({prompt_name!r}). Check prompts.csv.")
        prompt_names_by_pid[pid] = prompt_name

    return prompt_messages_by_pid, prompt_names_by_pid


def inject_test_text(messages: list[dict[str, str]], plain_text: str, placeholder: str = "[#TEST_TEXT]") -> list[dict[str, str]]:
    """
    Return a fresh copy of the message list with the dataset text inserted.
    If the placeholder is present, replace it.
    If not, append the text to the final user message.
    """
    filled_messages = deepcopy(messages)
    placeholder_found = False

    for message in filled_messages:
        if placeholder in message["content"]:
            message["content"] = message["content"].replace(placeholder, plain_text)
            placeholder_found = True

    if not placeholder_found:
        user_message_indices = [i for i, m in enumerate(filled_messages) if m["role"] == "user"]
        if not user_message_indices:
            raise ValueError("Prompt strategy does not contain a user message.")
        last_user_idx = user_message_indices[-1]
        filled_messages[last_user_idx]["content"] = (
            filled_messages[last_user_idx]["content"].rstrip() + "\n\n" + plain_text
        )

    return filled_messages