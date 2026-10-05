from ollama import ChatResponse, Client
from pandas import DataFrame
from tqdm.auto import tqdm
from src.prompt_engineering import inject_test_text
from typing import cast, Any
import time
import numpy as np

def prompt_ollama(ollama_client: Client, text: str, model: str = 'llama3.2:1b'):
    """
    Send a prompting request to Ollama.
    
    Args:
        text: Plain text to identify metaphors
        model: Model ID
    
    Returns:
        Tagged text with metaphors
    """
    messages = [
        {'role': 'system',
         'content': 'You are a linguistic expert trained in metaphor identification. When the user provides a text, follow this protocol:\n• Identify all metaphorical expressions.\n• Wrap each one in <Metaphor></Metaphor> tags.\n• Reproduce the rest of the text exactly as written.\n• Do not include any explanation, commentary, or extra content in this message.'},
        {'role': 'user',
         'content': f'Can you please identify and tag the metaphors in the following text?\n{text}'}
    ]
    try:
        response: ChatResponse = ollama_client.chat(model=model, messages=messages) # pyright: ignore[reportUnknownMemberType]
        return response.message.content
    except Exception as e:
        print(f'Error connecting to Ollama: {e}')
        print('Make sure Ollama service is running!')
    return None



def run_all_prompts_on_dataset_ollama(
    dataset: DataFrame,
    prompt_messages_by_pid: dict[int, list[dict[str, str]]],
    prompt_names_by_pid: dict[int, str],
    ollama_client: Client,
    model_name: str,
    num_repeats: int = 2,
    sleep_seconds: float = 0.0,
):
    '''This function was not there in the original notebooks.
    It was written by mimicking the same function in openai_utils.py'''
    rows: list[dict[str, Any]] = []

    if not all(isinstance(idx, int) for idx in dataset.index):
        raise TypeError("dataset must have a plain integer RangeIndex — call .reset_index(drop=True) before passing it in.")

    for pid, messages in tqdm(prompt_messages_by_pid.items(), desc="Prompt strategies"):
        prompt_name = prompt_names_by_pid[pid]

        for repeat_number in range(1, num_repeats + 1):
            for row_idx, row in tqdm(
                dataset.iterrows(), total=len(dataset),
                desc=f"Samples for prompt {pid} (run {repeat_number}/{num_repeats})",
                leave=False,
            ):
                plain_text = row["plain"]
                filled_messages = inject_test_text(messages, plain_text)

                try:
                    response = ollama_client.chat(model=model_name, messages=filled_messages)  # pyright: ignore[reportUnknownMemberType]
                    llm_output = response.message.content
                    error_message = None
                    status = "ok"
                except Exception as e:
                    llm_output = None
                    error_message = str(e)
                    status = "api_error"

                dataset_index = cast(int, row_idx)
                rows.append({
                    "prompt_id": pid,
                    "prompt_name": prompt_name,
                    "repeat_number": repeat_number,
                    "dataset_index": dataset_index,
                    "textid": row["textid"],
                    "model_name": model_name,
                    "plain": plain_text,
                    "gold_metaphor_tagged_text": row["metaphor_tagged_text"],
                    "llm_output": llm_output,
                    "prompt_tokens": np.nan,
                    "completion_tokens": np.nan,
                    "total_tokens": np.nan,
                    "estimated_cost_usd": np.nan,
                    "status": status,
                    "error_message": error_message,
                })

                if sleep_seconds:
                    time.sleep(sleep_seconds)

    return DataFrame(rows)