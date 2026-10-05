import numpy as np
from openai import OpenAI
from pandas import DataFrame
from copy import deepcopy
import time
from tqdm.auto import tqdm
from src.openai_utils import call_openai_chat, estimate_request_cost_usd
from typing import Any, cast
from src.langchain_utils import retrieve_codebook_context, build_codebook_rag_messages
from langchain_community.vectorstores.sklearn import SKLearnVectorStore
from ollama import Client

'''This RAG pipeline is based on the RAG WEB notebook, which is NOT how RAG was done in the paper.
This is more like a dynamic few-shot strategy, in which the 'best' examples are retrieved via cosine similarity and added to the prompt.
As such, this belongs more to the realm of prompt engineering, than to the RAG proper.'''

def get_embeddings_batch(texts: list[str], client: OpenAI, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Embed a list of texts in a single API call.
    Returns a 2-D NumPy array of shape (len(texts), embedding_dim).
    """
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
    return np.array(vectors, dtype=np.float32)

def normalise_and_save_embeddings(train_embeddings: np.ndarray,
                                  embeddings_path: str = "outputs/train_embeddings.npy") -> np.ndarray:
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_embeddings_normed = train_embeddings / np.where(norms == 0, 1, norms)

    np.save(embeddings_path, train_embeddings_normed)

    print(f"Embeddings shape: {train_embeddings_normed.shape}")
    print(f"Saved to: {embeddings_path}")
    return train_embeddings_normed

def retrieve_top_k(
    query_text: str,
    train_pool: DataFrame,
    train_embeddings_normed: np.ndarray,
    client: OpenAI,
    embedding_model: str,
    top_k: int = 3,
    ) -> DataFrame:
    """
    Embed the query text and return the top_k most similar rows from train_pool.
    """
    response = client.embeddings.create(model=embedding_model, input=[query_text])
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)
    query_vec /= np.linalg.norm(query_vec) if np.linalg.norm(query_vec) > 0 else 1.0

    similarities = train_embeddings_normed @ query_vec
    top_indices  = np.argsort(similarities)[::-1][:top_k]
    return train_pool.iloc[top_indices].reset_index(drop=True)


def build_rag_messages(
    base_messages: list[dict[str, str]],
    retrieved_rows: DataFrame,
    test_plain_text: str,
    placeholder: str = "[#TEST_TEXT]",
) -> list[dict[str, str]]:
    """
    Build a few-shot RAG prompt by:
    1. Taking all messages from base_messages up to (but not including) the final user turn.
    2. Inserting one user/assistant pair for each retrieved example.
    3. Appending the final user message with the test text substituted in.

    If no [#TEST_TEXT] placeholder is found, the test text is appended to the last user message.
    """
    messages = deepcopy(base_messages)

    # Find the last user message index — this will become the final query.
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not user_indices:
        raise ValueError("Base prompt contains no user message.")
    last_user_idx = user_indices[-1]

    # Substitute the test text into the final user message.
    final_user_msg = deepcopy(messages[last_user_idx])
    if placeholder in final_user_msg["content"]:
        final_user_msg["content"] = final_user_msg["content"].replace(placeholder, test_plain_text)
    else:
        final_user_msg["content"] = final_user_msg["content"].rstrip() + "\n\n" + test_plain_text

    # Build prefix: everything before the final user message.
    prefix = messages[:last_user_idx]

    # Build few-shot block: one user/assistant pair per retrieved example.
    few_shot_turns: list[dict[str, str]] = []
    for _, ex_row in retrieved_rows.iterrows():
        few_shot_turns.append({"role": "user",      "content": ex_row["plain"]})
        few_shot_turns.append({"role": "assistant", "content": ex_row["metaphor_tagged_text"]})

    return prefix + few_shot_turns + [final_user_msg]

def run_rag_on_dataset(
    dataset: DataFrame,
    train_pool: DataFrame,
    train_embeddings_normed: np.ndarray,
    prompt_messages_by_pid: dict[int, list[dict[str, str]]],
    prompt_names_by_pid: dict[int, str],
    client: OpenAI,
    chat_model: str = "gpt-5-mini",
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 3,
    num_repeats: int = 2,
    sleep_seconds: float = 0.2,
) -> DataFrame:
    """
    For every prompt strategy, retrieve similar examples for each test row,
    build the RAG prompt, call the API, and collect results.
    """
    rows: list[dict[str, Any]] = []

    for pid, base_messages in tqdm(prompt_messages_by_pid.items(), desc="Prompt strategies"):
        prompt_name = prompt_names_by_pid[pid]

        for repeat_number in range(1, num_repeats + 1):
            for row_idx, row in tqdm(
                dataset.iterrows(),
                total=len(dataset),
                desc=f"Samples for prompt {pid} (run {repeat_number}/{num_repeats})",
                leave=False,
            ):
                plain_text = row["plain"]

                try:
                    retrieved_rows = retrieve_top_k(
                        query_text=plain_text,
                        train_pool=train_pool,
                        train_embeddings_normed=train_embeddings_normed,
                        client=client,
                        embedding_model=embedding_model,
                        top_k=top_k,
                    )

                    rag_messages = build_rag_messages(
                        base_messages=base_messages,
                        retrieved_rows=retrieved_rows,
                        test_plain_text=plain_text,
                    )

                    response_info     = call_openai_chat(client, chat_model, rag_messages, temperature=0.0)
                    llm_output        = response_info.llm_output
                    prompt_tokens     = response_info.prompt_tokens
                    completion_tokens = response_info.completion_tokens
                    total_tokens      = response_info.total_tokens
                    estimated_cost_usd = estimate_request_cost_usd(
                        prompt_tokens, completion_tokens, chat_model
                    )
                    retrieved_textids  = ",".join(str(r) for r in retrieved_rows["textid"].tolist())
                    error_message = None
                    status        = "ok"

                except Exception as e:
                    llm_output         = None
                    prompt_tokens      = np.nan
                    completion_tokens  = np.nan
                    total_tokens       = np.nan
                    estimated_cost_usd = np.nan
                    retrieved_textids  = None
                    error_message      = str(e)
                    status             = "api_error"

                rows.append({
                    "prompt_id":               pid,
                    "prompt_name":             prompt_name,
                    "repeat_number":           repeat_number,
                    "dataset_index":           row_idx,
                    "textid":                  row["textid"],
                    "model_name":              chat_model,
                    "top_k":                   top_k,
                    "retrieved_textids":       retrieved_textids,
                    "plain":                   plain_text,
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

'''This section of the code is devoted to the actual RAG strategy that is used in the paper.
It is based on the retrieval of chunks of the codebook. As it is, it doesn't make much sense, and unsurprisingly it turns out to be the worst strategy.'''

def run_codebook_rag_on_dataset(
    dataset: DataFrame,
    vectorstore: SKLearnVectorStore,
    client: OpenAI,
    chat_model: str = "gpt-4.1-mini-2025-04-14",
    top_k: int = 1,
    num_repeats: int = 5,
    sleep_seconds: float = 0.2,
) -> DataFrame:
    rows: list[dict[str, Any]] = []

    if not all(isinstance(idx, int) for idx in dataset.index):
        raise TypeError("dataset must have a plain integer RangeIndex — call .reset_index(drop=True) before passing it in.")

    for repeat_number in range(1, num_repeats + 1):
        for row_idx, row in tqdm(
            dataset.iterrows(), total=len(dataset),
            desc=f"RAG (codebook) run {repeat_number}/{num_repeats}",
        ):
            plain_text = row["plain"]

            try:
                context = retrieve_codebook_context(vectorstore, plain_text, top_k=top_k)
                rag_messages = build_codebook_rag_messages(context, plain_text)

                response_info      = call_openai_chat(client, chat_model, rag_messages, temperature=0.0)
                llm_output         = response_info.llm_output
                prompt_tokens      = response_info.prompt_tokens
                completion_tokens  = response_info.completion_tokens
                total_tokens       = response_info.total_tokens
                estimated_cost_usd = estimate_request_cost_usd(prompt_tokens, completion_tokens, chat_model)
                error_message = None
                status = "ok"
            except Exception as e:
                llm_output = None
                prompt_tokens = completion_tokens = total_tokens = estimated_cost_usd = np.nan
                error_message = str(e)
                status = "api_error"

            dataset_index = cast(int, row_idx)
            rows.append({
                "repeat_number":            repeat_number,
                "dataset_index":            dataset_index,
                "textid":                   row["textid"],
                "model_name":               chat_model,
                "top_k":                    top_k,
                "plain":                    plain_text,
                "gold_metaphor_tagged_text": row["metaphor_tagged_text"],
                "llm_output":               llm_output,
                "prompt_tokens":            prompt_tokens,
                "completion_tokens":        completion_tokens,
                "total_tokens":             total_tokens,
                "estimated_cost_usd":       estimated_cost_usd,
                "status":                   status,
                "error_message":            error_message,
            })

            time.sleep(sleep_seconds)

    return DataFrame(rows)

def run_codebook_rag_on_dataset_ollama(
    dataset: DataFrame,
    vectorstore: SKLearnVectorStore,
    ollama_client: Client,
    model_name: str,
    top_k: int = 1,
    num_repeats: int = 5,
    sleep_seconds: float = 0.0,
) -> DataFrame:
    rows: list[dict[str, Any]] = []

    if not all(isinstance(idx, int) for idx in dataset.index):
        raise TypeError("dataset must have a plain integer RangeIndex — call .reset_index(drop=True) before passing it in.")

    for repeat_number in range(1, num_repeats + 1):
        for row_idx, row in tqdm(
            dataset.iterrows(), total=len(dataset),
            desc=f"RAG (codebook) run {repeat_number}/{num_repeats}",
        ):
            plain_text = row["plain"]

            try:
                context = retrieve_codebook_context(vectorstore, plain_text, top_k=top_k)
                rag_messages = build_codebook_rag_messages(context, plain_text)
                response = ollama_client.chat(model=model_name, messages=rag_messages)  # pyright: ignore[reportUnknownMemberType]
                llm_output = response.message.content
                error_message = None
                status = "ok"
            except Exception as e:
                llm_output = None
                error_message = str(e)
                status = "api_error"

            dataset_index = cast(int, row_idx)
            rows.append({
                "repeat_number":            repeat_number,
                "dataset_index":            dataset_index,
                "textid":                   row["textid"],
                "model_name":               model_name,
                "top_k":                    top_k,
                "plain":                    plain_text,
                "gold_metaphor_tagged_text": row["metaphor_tagged_text"],
                "llm_output":               llm_output,
                "prompt_tokens":            np.nan,
                "completion_tokens":        np.nan,
                "total_tokens":             np.nan,
                "estimated_cost_usd":       np.nan,
                "status":                   status,
                "error_message":            error_message,
            })

            if sleep_seconds:
                time.sleep(sleep_seconds)

    return DataFrame(rows)