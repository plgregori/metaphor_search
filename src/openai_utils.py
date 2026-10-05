from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pandas import DataFrame, Series, isna, read_csv
import json
import numpy as np
from pydantic import BaseModel
from tqdm.auto import tqdm
import time
from typing import cast
from pathlib import Path
from io import BytesIO
import re
from src.prompt_engineering import inject_test_text

MODEL_PRICING_PER_1M_TOKENS = {
    # --- Core GPT-5 family ---
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},

    # --- GPT-5.1 series ---
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5.1-mini": {"input": 0.25, "output": 2.00},

    # --- GPT-5.2 series ---
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-mini": {"input": 0.25, "output": 2.00},

    # --- GPT-5.3 series ---
    "gpt-5.3": {"input": 1.75, "output": 14.00},

    # --- GPT-5.4 (latest flagship) ---
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.00},

    # --- Chat variants ---
    "gpt-5-chat": {"input": 1.25, "output": 10.00},
    "gpt-5.1-chat": {"input": 1.25, "output": 10.00},
    "gpt-5.2-chat": {"input": 1.75, "output": 14.00},
    "gpt-5.3-chat": {"input": 1.75, "output": 14.00},

}

MODEL_PRICE_ALIASES = {
    # --- Core ---
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",

    # --- Versioned snapshots ---
    "gpt-5-2025-08-01": "gpt-5",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1-2025-09-01": "gpt-5.1",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-2025-12-11": "gpt-5.2",
    "gpt-5.3": "gpt-5.3",
    "gpt-5.4": "gpt-5.4",

    # --- Mini/Nano ---
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.4-nano": "gpt-5.4-nano",

    # --- Chat aliases ---
    "gpt-5-chat-latest": "gpt-5-chat",
    "gpt-5.1-chat-latest": "gpt-5.1-chat",
    "gpt-5.2-chat-latest": "gpt-5.2-chat",
    "gpt-5.3-chat-latest": "gpt-5.3-chat",

}

class OpenAIOutput(BaseModel):
    llm_output: str | None
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float


def _count_ok(s: Series) -> int:
    return int((s == "ok").sum())

def _sum_cost(s: Series) -> float:
    return s.sum(min_count=1)


_DATE_SUFFIX_PATTERN = re.compile(r"-\d{4}-\d{2}-\d{2}$")

def normalise_model_name_for_pricing(model_name: str) -> str:
    if model_name.startswith("ft:"):
        parts = model_name.split(":")
        model_name = parts[1] if len(parts) > 1 else model_name

    if model_name in MODEL_PRICE_ALIASES:
        return MODEL_PRICE_ALIASES[model_name]

    stripped = _DATE_SUFFIX_PATTERN.sub("", model_name)
    if stripped in MODEL_PRICE_ALIASES:
        return MODEL_PRICE_ALIASES[stripped]
    if stripped in MODEL_PRICING_PER_1M_TOKENS:
        return stripped

    return model_name

def estimate_request_cost_usd(
    prompt_tokens: int | float | None,
    completion_tokens: int | float | None,
    model_name: str,
) -> float:
    """
    Estimate request cost in USD from token counts and the model name.
    Returns NaN if the model is not in the pricing table.
    """
    pricing_model_name = normalise_model_name_for_pricing(model_name)
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(pricing_model_name)

    if pricing is None:
        return np.nan

    prompt_tokens = 0 if isna(prompt_tokens) else prompt_tokens
    completion_tokens = 0 if isna(completion_tokens) else completion_tokens

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return float(input_cost + output_cost)

def openai_initializer(api_key: str) -> OpenAI:
    client_openai = OpenAI(api_key=api_key)
    print('✓ OpenAI client initialized')
    return client_openai


def prompt_openai_simple(client: OpenAI, text: str, model: str= 'gpt-4.1-mini-2025-04-14'):
    """
    Send a prompting request to OpenAI.
    
    Args:
        client: OpenAI client
        text: Plain text to identify metaphors
        model: Model ID
    
    Returns:
        Tagged text with metaphors
    """
    messages: list[ChatCompletionMessageParam] = [
        {'role': 'system',
         'content': 'You are a linguistic expert trained in metaphor identification. When the user provides a text, follow this protocol:\n• Identify all metaphorical expressions.\n• Wrap each one in <Metaphor></Metaphor> tags.\n• Reproduce the rest of the text exactly as written.\n• Do not include any explanation, commentary, or extra content in this message.'},
        {'role': 'user',
         'content': f'Can you please identify and tag the metaphors in the following text?\n{text}'}
        ]
    try:
        response = client.chat.completions.create(model=model, messages=messages, n=1)
        return response.choices[0].message.content
    except Exception as e:
        print(f'Error: {e}')
        return None
    
def prepare_finetuning_data_openai(dataset: DataFrame, train_ratio: float = 0.8, seed: int = 1) -> tuple[DataFrame, DataFrame, str]:
    """
    Prepare data for OpenAI fine-tuning (JSONL format).
    
    Args:
        dataset: DataFrame with 'plain' and tagged text columns
        train_ratio: Train/test split ratio
        seed: Random seed
    
    Returns:
        train_df, test_df, jsonl_path
    """
    # Split data
    train_df = dataset.sample(frac=train_ratio, random_state=seed)
    test_df = dataset.drop(index=train_df.index)
    
    # Prepare JSONL
    user_msg = 'Can you please identify and tag the metaphors in the following text? '
    json_lines: list[str] = []
    
    for idx in range(len(train_df)):
        raw_text = train_df.iloc[idx]['plain'].replace('\\n', ' ')
        tagged_text = train_df.iloc[idx]['metaphor_tagged_text'].replace('\\n', ' ')
        this_chat: dict[str, list[dict[str,str]]] = {
            'messages': [
                {'role': 'user', 'content': user_msg + '\n' + raw_text},
                {'role': 'assistant', 'content': tagged_text}
            ]
        }
        json_lines.append(json.dumps(this_chat))
    
    jsonl_path = f'outputs/ft_tr{train_ratio}_s{seed}.jsonl'
    with open(jsonl_path, 'w') as f:
        f.write('\n'.join(json_lines))

    print(f'Fine-tuning data prepared:')
    print(f'  Train set: {len(train_df)} samples')
    print(f'  Test set: {len(test_df)} samples')
    print(f'  JSONL file: {jsonl_path}')
    
    return train_df, test_df, jsonl_path

def load_and_check_prompt_data_openai(dataset_path: str, prompts_path: str, row_limit: int | None = None) -> tuple[DataFrame, DataFrame]:
    '''Row limit set to None processes the full dataset. Set to a small integer to run quick experiments.'''
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

    print(f"Dataset rows loaded:     {len(dataset_df)}")
    print(f"Prompt rows loaded:      {len(prompts_df)}")
    print(f"Unique prompt strategies:{prompts_df['pid'].nunique()}")
    return dataset_df, prompts_df

def create_fine_tuning_openai(client_openai: OpenAI, jsonl_path: str, model: str= 'gpt-4.1-mini-2025-04-14' ):
    # Upload file
    with open(jsonl_path, 'rb') as f:
        fileinfo = client_openai.files.create(file=f, purpose='fine-tune')
     
    # Create fine-tuning job
    ft_job = client_openai.fine_tuning.jobs.create(
        training_file=fileinfo.id,
        model=model,
        suffix='metaphor_ft'
    )
    print(f'Fine-tuning job created: {ft_job.id}')

def prepare_finetuning_data_openai_bis(dataset: DataFrame, train_fraction: float = 0.8, random_seed: int = 42, min_train_examples: int = 10) -> tuple[DataFrame, DataFrame]:
    train_df = dataset.sample(frac=train_fraction, random_state=random_seed)
    test_df  = dataset.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    if len(train_df) < min_train_examples:
        raise ValueError(
            f"Training split has only {len(train_df)} rows. "
            f"The OpenAI fine-tuning API requires at least {min_train_examples}. "
            "Increase TRAIN_FRACTION or reduce ROW_LIMIT."
        )

    print(f"Training examples: {len(train_df)}")
    print(f"Test examples:     {len(test_df)}")
    return train_df, test_df

def extract_system_instruction(prompts_table: DataFrame) -> str:
    sorted_by_pid = prompts_table.sort_values("pid")
    first_pid = cast(object, sorted_by_pid["pid"].iloc[0])
    first_strategy = prompts_table[prompts_table["pid"] == first_pid].sort_values("cid")
    system_rows = first_strategy[first_strategy["role"] == "system"]
    if not system_rows.empty:
        return " ".join(system_rows["content"].tolist()).strip()
    fallback = (
        "You are a linguistic annotator. "
        "Read the text and wrap every metaphorical expression in <Metaphor>...</Metaphor> tags. "
        "Return the full text with tags inserted and nothing else."
    )
    print("Warning: no system row found in prompts.csv. Using fallback instruction.")
    return fallback

def save_training_jsonl(df: DataFrame, system_instruction: str, output_path: str):
    lines: list[str] = []
    for _, row in df.iterrows():
        example: dict[str, list[dict[str,str]]] = {
            "messages": [
                {"role": "system",    "content": system_instruction},
                {"role": "user",      "content": row["plain"]},
                {"role": "assistant", "content": row["metaphor_tagged_text"]},
            ]
        }
        lines.append(json.dumps(example, ensure_ascii=False))
    train_jsonl_str = "\n".join(lines)
    local_jsonl_path = Path(output_path)
    local_jsonl_path.write_text(train_jsonl_str, encoding="utf-8")
    print(f"Training JSONL saved to: {local_jsonl_path.resolve()}")
    print(f"Lines in JSONL: {train_jsonl_str.count(chr(10)) + 1}")
    print("\nFirst example preview:")
    print(train_jsonl_str.split("\n")[0][:300])

def prepare_rag_context() -> str:
    """
    Prepare RAG context from metaphor protocol.
    """
    rag_context = """METAPHOR IDENTIFICATION PROTOCOL:
    
    A metaphor is a figure of speech that directly compares two different things 
    without using 'like' or 'as'. It implies that something IS something else.
    
    Key characteristics:
    1. Two distinct entities being compared
    2. Implicit comparison (no 'like' or 'as')
    3. Creates new meaning through the comparison
    
    Examples:
    - 'Time is money' (time and money are compared)
    - 'Life is a journey' (life and journey)
    - 'The world is a stage' (world and stage)
    
    When identifying metaphors, tag them with <Metaphor></Metaphor> tags.
    """
    return rag_context

def prompt_openai_rag(client: OpenAI, text: str, context: str, model: str = 'gpt-4.1-mini-2025-04-14') -> None | str:
    """
    RAG-based prompting with OpenAI.
    """
    messages: list[ChatCompletionMessageParam]  = [
        {'role': 'system',
         'content': 'You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don\'t know the answer, just say you don\'t know. DO NOT try to make up an answer.\n' + context},
        {'role': 'system',
         'content': 'You are a linguistic expert trained in metaphor identification. When the user provides a text, follow this protocol:\n• Identify all metaphorical expressions.\n• Wrap each one in <Metaphor></Metaphor> tags.\n• Reproduce the rest of the text exactly as written.\n• Do not include any explanation, commentary, or extra content.'},
        {'role': 'user',
         'content': f'Can you please identify and tag the metaphors in the following text?\n{text}'}
    ]
    try:
        response = client.chat.completions.create(model=model, messages=messages, n=1)
        return response.choices[0].message.content
    except Exception as e:
        print(f'Error: {e}')
        return None
    

def call_openai_chat(client: OpenAI, model_name: str, messages: list[dict[str, str]], temperature: float = 0.0):
    """
    Send one chat-style request and return a dictionary with the model text and token usage.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=cast(list[ChatCompletionMessageParam], messages),
        temperature=temperature,
    )

    output_text = response.choices[0].message.content
    output_text = output_text.strip() if isinstance(output_text, str) else output_text

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", np.nan) if usage is not None else np.nan
    completion_tokens = getattr(usage, "completion_tokens", np.nan) if usage is not None else np.nan
    total_tokens = getattr(usage, "total_tokens", np.nan) if usage is not None else np.nan

    return OpenAIOutput(
        llm_output = output_text,
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        total_tokens = total_tokens)

def run_fine_tuned_model_on_dataset(
    dataset: DataFrame,
    system_instruction: str,
    client: OpenAI,
    model_name: str,
    num_repeats: int = 2,
    sleep_seconds: float = 0.2,
):
    """
    Run the fine-tuned model on every row of the dataset, repeating each
    inference num_repeats times. Returns a DataFrame with one row per
    text × repeat.
    """
    rows: list[dict[str, int | None | str | float]] = []

    if not all(isinstance(idx, int) for idx in dataset.index):
        raise TypeError("dataset must have a plain integer RangeIndex — call .reset_index(drop=True) before passing it in.")

    for repeat_number in range(1, num_repeats + 1):
        for row_idx, row in tqdm(
            dataset.iterrows(),
            total=len(dataset),
            desc=f"Inference run {repeat_number}/{num_repeats}",
        ):
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": row["plain"]},
            ]

            try:
                response_info     = call_openai_chat(client, model_name, messages, temperature=0.0)
                llm_output        = response_info.llm_output
                prompt_tokens     = response_info.prompt_tokens
                completion_tokens = response_info.completion_tokens
                total_tokens      = response_info.total_tokens
                estimated_cost_usd = estimate_request_cost_usd(
                    prompt_tokens, completion_tokens, model_name
                )
                error_message = None
                status        = "ok"
            except Exception as e:
                llm_output         = None
                prompt_tokens      = np.nan
                completion_tokens  = np.nan
                total_tokens       = np.nan
                estimated_cost_usd = np.nan
                error_message      = str(e)
                status             = "api_error"
            dataset_index = cast(int, row_idx)
            rows.append({
                "model_name":              model_name,
                "repeat_number":           repeat_number,
                "dataset_index":           dataset_index,
                "textid":                  row["textid"],
                "plain":                   row["plain"],
                "gold_metaphor_tagged_text": row["metaphor_tagged_text"],
                "llm_output":              llm_output,
                "prompt_tokens":           prompt_tokens,
                "completion_tokens":       completion_tokens,
                "total_tokens":            total_tokens,
                "estimated_cost_usd":      estimated_cost_usd,
                "status":                  status,
                "error_message":           error_message,
            })

            time.sleep(sleep_seconds)

    return DataFrame(rows)

def run_all_prompts_on_dataset(
    dataset: DataFrame,
    prompt_messages_by_pid: dict[int, list[dict[str, str]]],
    prompt_names_by_pid: dict[int, str],
    client: OpenAI,
    model_name: str,
    num_repeats: int = 2,
    sleep_seconds: float = 0.2,
):
    rows: list[dict[str, int | None | str | float]] = []

    if not all(isinstance(idx, int) for idx in dataset.index):
        raise TypeError("dataset must have a plain integer RangeIndex — call .reset_index(drop=True) before passing it in.")

    for pid, messages in tqdm(prompt_messages_by_pid.items(), desc="Prompt strategies"):
        prompt_name = prompt_names_by_pid[pid]

        for repeat_number in range(1, num_repeats + 1):
            for row_idx, row in tqdm(
                dataset.iterrows(),
                total=len(dataset),
                desc=f"Samples for prompt {pid} (run {repeat_number}/{num_repeats})",
                leave=False,
            ):
                plain_text = row["plain"]
                filled_messages = inject_test_text(messages, plain_text)

                try:
                    response_info = call_openai_chat(
                        client=client,
                        model_name=model_name,
                        messages=filled_messages,
                        temperature=0.0,
                    )
                    llm_output = response_info.llm_output
                    prompt_tokens = response_info.prompt_tokens
                    completion_tokens = response_info.completion_tokens
                    total_tokens = response_info.total_tokens
                    estimated_cost_usd = estimate_request_cost_usd(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        model_name=model_name,
                    )
                    error_message = None
                    status = "ok"
                except Exception as e:
                    llm_output = None
                    prompt_tokens = np.nan
                    completion_tokens = np.nan
                    total_tokens = np.nan
                    estimated_cost_usd = np.nan
                    error_message = str(e)
                    status = "api_error"
                dataset_index = cast(int, row_idx)
                rows.append(
                    {
                        "prompt_id": pid,
                        "prompt_name": prompt_name,
                        "repeat_number": repeat_number,
                        "dataset_index": dataset_index,
                        "textid": row["textid"],
                        "model_name": model_name,
                        "plain": plain_text,
                        "gold_metaphor_tagged_text": row["metaphor_tagged_text"],
                        "llm_output": llm_output,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "estimated_cost_usd": estimated_cost_usd,
                        "status": status,
                        "error_message": error_message,
                    }
                )

                time.sleep(sleep_seconds)

    return DataFrame(rows)

def summarise_experiment_costs(results_table: DataFrame):
    """
    Summarise token usage and estimated cost by model.
    """
    if results_table.empty:
        return DataFrame()

    summary = (
        results_table.groupby("model_name", dropna=False)
        .agg(
            num_requests=("model_name", "size"),
            successful_requests=("status", _count_ok),
            prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            estimated_cost_usd=("estimated_cost_usd", _sum_cost),
        )
        .reset_index()
    )

    return summary

def submit_finetuning_jsonl(client: OpenAI,
                            jsonl_path: str,
                            base_model_name: str = "gpt-5-mini",
                            fine_tuned_model_suffix: str = "metaphor",
                            n_epochs: int | None = None,
                            model_id_path: str = "outputs/fine_tuned_model_id.txt") -> str | None:
    '''The number of epochs left to None lets OpenAI choose automatically'''
    train_jsonl_str = Path(jsonl_path).read_text(encoding="utf-8")
    upload_response = client.files.create(
    file=(Path(jsonl_path).name, BytesIO(train_jsonl_str.encode("utf-8")), "application/jsonl"), purpose="fine-tune")
    training_file_id = upload_response.id
    print(f"Uploaded training file. File ID: {training_file_id}")
    print("Waiting for file to be processed...", end="", flush=True)
    while True:
        file_status = client.files.retrieve(training_file_id).status
        if file_status == "processed":
            print(" done.")
            break
        elif file_status == "error":
            raise RuntimeError(f"File processing failed. File ID: {training_file_id}")
        print(".", end="", flush=True)
        time.sleep(5)

    print(f"File status: {file_status}")

    
    ft_job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=base_model_name,
        hyperparameters={"n_epochs": n_epochs} if n_epochs is not None else {"n_epochs": "auto"},
        suffix=fine_tuned_model_suffix,
    )

    ft_job_id = ft_job.id
    print(f"Fine-tuning job submitted.")
    print(f"  Job ID:  {ft_job_id}")
    print(f"  Status:  {ft_job.status}")
    print(f"  Model:   {ft_job.model}")

    POLL_INTERVAL_SECONDS = 30
    TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

    print(f"Polling fine-tuning job {ft_job_id}...")
    while True:
        job_info = client.fine_tuning.jobs.retrieve(ft_job_id)
        status   = job_info.status
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {status}")
        if status in TERMINAL_STATUSES:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    if status == "succeeded":
        fine_tuned_model_name = job_info.fine_tuned_model
        print(f"\nFine-tuning succeeded!")
        print(f"Fine-tuned model ID: {fine_tuned_model_name}")
        Path(model_id_path).write_text(str(fine_tuned_model_name), encoding="utf-8")
        print(f"Fine-tuned model ID saved to: {model_id_path}")
        print(f"Model ID: {fine_tuned_model_name}")
        return fine_tuned_model_name
    elif status == "failed":
        error_info = getattr(job_info, "error", None)
        raise RuntimeError(
            f"Fine-tuning job failed.\n"
            f"Job ID: {ft_job_id}\n"
            f"Error: {error_info}"
        )
    else:
        raise RuntimeError(f"Fine-tuning job ended with unexpected status: {status}")