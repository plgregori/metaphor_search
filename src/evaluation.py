from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix
import numpy as np
from numpy import ndarray
import re
from pandas import DataFrame, isna, Series
from pydantic import BaseModel, ConfigDict

def needleman_wunsch_indices(a: list[str],
                     b: list[str],
                     match_score:int = 1,
                     mismatch_penalty: int = -1,
                     gap_penalty: int = -3) -> list[tuple[int | None, int | None]]:
    """Classic Needleman–Wunsch global alignment for sequences of tokens. Outputs the aligned sequences of indices."""
    n, m = len(a), len(b)
    # Initialize score matrix
    score = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        score[i][0] = i * gap_penalty
    for j in range(1, m+1):
        score[0][j] = j * gap_penalty
    # Fill
    for i in range(1, n+1):
        for j in range(1, m+1):
            diag = score[i-1][j-1] + (match_score if a[i-1] == b[j-1] else mismatch_penalty)
            up = score[i-1][j] + gap_penalty
            left = score[i][j-1] + gap_penalty
            score[i][j] = max(diag, up, left)
    # Traceback
    aligned: list[tuple[int| None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i-1][j-1] + (match_score if a[i-1] == b[j-1] else mismatch_penalty):
            aligned.append((i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and score[i][j] == score[i-1][j] + gap_penalty:
            aligned.append((i-1, None))
            i -= 1
        else:
            aligned.append((None, j-1))
            j -= 1
    aligned.reverse()
    return aligned

def needleman_wunsch_tokens(a: list[str],
                             b: list[str],
                             match_score: int = 1,
                             mismatch_penalty: int = -1,
                             gap_penalty: int = -3) -> list[tuple[str | None, str | None]]:
    """Same alignment as needleman_wunsch_indices, expressed as token values instead of indices."""
    pairs = needleman_wunsch_indices(a, b, match_score, mismatch_penalty, gap_penalty)
    return [
        (a[i] if i is not None else None, b[j] if j is not None else None)
        for i, j in pairs
    ]

def do_tokenize(text: str) -> list[str]:

    text = text.replace(">","> ")
    text = text.replace("<"," <")
    tokens = text.split()
    return tokens

def xml_to_bin(xml: str, tag_name: str) -> tuple[list[int], list[str]]:

    tokens=do_tokenize(xml)

    y_bin: list[int] = []
    y_bin_ref: list[str] = []

    tag_switch = 0
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"

    for token in tokens:
        if(start_tag in token):
            tag_switch = 1
            continue
        if(end_tag in token):
            tag_switch = 0
            continue
        y_bin.append(tag_switch)
        y_bin_ref.append(token)

    return y_bin, y_bin_ref

class Praf(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    true_pred_disp: int
    true_align_disp: int
    precision: float
    recall: float
    accuracy: float
    f1: float
    confusion_matrix: ndarray
    y_true: list[int]
    y_pred: list[int]
    y_true_token: list[str]
    y_pred_token: list[str]

class EvalResults(BaseModel):
    precision: float
    recall: float
    accuracy: float
    f1: float
    num_samples: int

def do_praf(xml_true: str, xml_pred: str, tag_name: str) -> Praf:

    praf_class: Praf
    true_bin, true_bin_ref = xml_to_bin(xml_true, tag_name)
    pred_bin, pred_bin_ref = xml_to_bin(xml_pred, tag_name)

    pair_aligned: list[tuple[int | None,int | None]] = needleman_wunsch_indices(true_bin_ref, pred_bin_ref)
    true_bin_aligned: list[int] = []
    pred_bin_aligned: list[int] = []
    true_bin_ref_aligned: list[str] = []
    pred_bin_ref_aligned: list[str] = []

    for true_i, pred_i in pair_aligned:
        if true_i is None:
            true_bin_aligned.append(0)
            true_bin_ref_aligned.append("[NOM]")
        else:
            true_bin_aligned.append(true_bin[true_i])
            true_bin_ref_aligned.append(true_bin_ref[true_i])
        if pred_i is None:
            pred_bin_aligned.append(0)
            pred_bin_ref_aligned.append("[NOM]")
        else:
            pred_bin_aligned.append(pred_bin[pred_i])
            pred_bin_ref_aligned.append(pred_bin_ref[pred_i])
    precision = float(precision_score(true_bin_aligned,pred_bin_aligned,average="macro"))
    recall = float(recall_score(true_bin_aligned,pred_bin_aligned,average="macro"))
    f1 = float(f1_score(true_bin_aligned,pred_bin_aligned,average="macro"))
    accuracy = float(accuracy_score(true_bin_aligned,pred_bin_aligned))
    praf_class = Praf(true_pred_disp = len(true_bin)-len(pred_bin),
                      true_align_disp = len(true_bin)-len(true_bin_aligned),
                      precision = precision,
                      recall = recall,
                      accuracy = accuracy,
                      f1 = f1,
                      confusion_matrix = confusion_matrix(true_bin_aligned,pred_bin_aligned,labels=[0,1]),
                      y_true = true_bin_aligned,
                      y_pred = pred_bin_aligned,
                      y_true_token = true_bin_ref_aligned,
                      y_pred_token = pred_bin_ref_aligned)

    return praf_class

def parse_metaphor_tags(xml_text: str, tag_name: str = 'Metaphor'):
    """
    Extract text wrapped in metaphor tags.
    """
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = re.findall(pattern, xml_text, re.DOTALL)
    return matches

def evaluate_all_samples(dataset: DataFrame, predictions: list[str] | None = None) -> EvalResults | None:
    """
    Evaluate metaphor identification across all samples.

    Args:
        dataset: DataFrame with ground truth
        predictions: List of predictions (or use ground truth for perfect match)

    Returns:
        Dictionary with metrics
    """
    if predictions is None:
        # Use ground truth as prediction for demonstration
        predictions = dataset['metaphor_tagged_text'].tolist()

    all_results: list[Praf] = []

    for idx in range(len(dataset)):
        xml_true = dataset.iloc[idx]['metaphor_tagged_text']
        xml_pred = predictions[idx]

        try:
            result = do_praf(xml_true, xml_pred, 'Metaphor')
            all_results.append(result)
        except Exception as e:
            print(f'Error evaluating sample {idx}: {e}')
            continue

    if not all_results:
        print('No valid results to aggregate')
        return None

    # Aggregate metrics
    avg_precision = np.mean([r.precision for r in all_results])
    avg_recall = np.mean([r.recall for r in all_results])
    avg_accuracy = np.mean([r.accuracy for r in all_results])
    avg_f1 = np.mean([r.f1 for r in all_results])

    return EvalResults(
        precision = avg_precision,
        recall = avg_recall,
        accuracy = avg_accuracy,
        f1 = avg_f1,
        num_samples = len(all_results)
    )

def evaluate_predictions(results_table: DataFrame) -> DataFrame:

    _ROW_METRICS_COLUMNS = [
    "prompt_id", "prompt_name", "repeat_number", "dataset_index", "textid",
    "model_name", "top_k", "status", "estimated_cost_usd",
    "precision", "recall", "accuracy", "f1",
    "true_pred_disp", "true_align_disp", "evaluation_error",
]
    row_metrics: list[dict[str, str | float | None]] = []

    for _, row in results_table.iterrows():
        if isna(row["llm_output"]) or row["llm_output"] is None or str(row["llm_output"]).strip() == "":
            continue

        try:
            metrics = do_praf(
                xml_true=row["gold_metaphor_tagged_text"],
                xml_pred=row["llm_output"],
                tag_name="Metaphor",
            )

            row_metrics.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "prompt_name": row.get("prompt_name"),
                    "repeat_number": row["repeat_number"],
                    "dataset_index": row["dataset_index"],
                    "textid": row["textid"],
                    "model_name": row["model_name"],
                    "top_k": row.get("top_k"),
                    "status": row["status"],
                    "estimated_cost_usd": row["estimated_cost_usd"],
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "accuracy": metrics.accuracy,
                    "f1": metrics.f1,
                    "true_pred_disp": metrics.true_pred_disp,
                    "true_align_disp": metrics.true_align_disp
                }
            )
        except Exception as e:
            row_metrics.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "prompt_name": row.get("prompt_name"),
                    "repeat_number": row["repeat_number"],
                    "dataset_index": row["dataset_index"],
                    "textid": row["textid"],
                    "model_name": row["model_name"],
                    "top_k": row.get("top_k"),
                    "status": row["status"],
                    "estimated_cost_usd": row["estimated_cost_usd"],
                    "precision": np.nan,
                    "recall": np.nan,
                    "accuracy": np.nan,
                    "f1": np.nan,
                    "true_pred_disp": np.nan,
                    "true_align_disp": np.nan,
                    "evaluation_error": str(e),
                }
            )

    return DataFrame(row_metrics) if row_metrics else DataFrame(columns=_ROW_METRICS_COLUMNS)

def _no_metaphor_tag(cell: object) -> bool:
    METAPHOR_TAG_PATTERN = r"<Metaphor>.*?</Metaphor>"
    return not bool(re.search(METAPHOR_TAG_PATTERN, str(cell), re.DOTALL))

def identify_no_metaphors(results_table: DataFrame, missing_metaphor_csv_path: str) -> DataFrame:
    

    no_metaphor_mask: Series[bool] = (
        results_table["status"].eq("ok")
        & results_table["llm_output"].apply(_no_metaphor_tag)
    )

    missing_metaphor_tags_df = results_table[no_metaphor_mask].copy()

    for optional_column in ("prompt_id", "prompt_name"):
        if optional_column not in missing_metaphor_tags_df.columns:
            missing_metaphor_tags_df[optional_column] = None

    missing_metaphor_summary_df = missing_metaphor_tags_df[
        [
            "prompt_id",
            "prompt_name",
            "repeat_number",
            "dataset_index",
            "textid",
            "model_name",
            "plain",
            "status",
            "error_message",
        ]
    ].copy()

    missing_metaphor_summary_df.insert(0, "csv_row_index", missing_metaphor_tags_df.index)
    missing_metaphor_summary_df.to_csv(missing_metaphor_csv_path, index=False)
    return missing_metaphor_summary_df

def _sum_or_nan(s: "Series[float]") -> float:
    return s.sum(min_count=1)

def summarize_scores(row_level_metrics_table: DataFrame) -> DataFrame:

    if row_level_metrics_table.empty:
        return DataFrame()
    
    prompt_level_metrics_df = (
        row_level_metrics_table.groupby(["prompt_id", "prompt_name", "model_name", "top_k"], dropna=False)
        .agg(
            runs_scored=("prompt_id", "size"),
            unique_texts=("textid", "nunique"),
            repeats_completed=("repeat_number", "nunique"),
            avg_precision=("precision", "mean"),
            avg_recall=("recall", "mean"),
            avg_accuracy=("accuracy", "mean"),
            avg_f1=("f1", "mean"),
            total_estimated_cost_usd=("estimated_cost_usd", _sum_or_nan),
        )
        .reset_index()
        .sort_values(["avg_f1", "avg_precision", "avg_recall"], ascending=False)
    )
    return prompt_level_metrics_df

def save_evaluation(row_level_metrics_table: DataFrame, row_level_path: str, prompt_level_path: str):
    row_level_metrics_table.to_csv(row_level_path, index=False)
    prompt_level_metrics_table = summarize_scores(row_level_metrics_table)
    prompt_level_metrics_table.to_csv(prompt_level_path, index=False)
    print(f"Saved row-level metrics to: {row_level_path}")
    print(f"Saved prompt-level metrics to: {prompt_level_path}")
