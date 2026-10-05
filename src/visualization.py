from pydantic import BaseModel
from html import escape
from src.evaluation import xml_to_bin, needleman_wunsch_indices
from pathlib import Path


class Token (BaseModel):
    text: str
    is_metaphor: bool

# Colors for categories
COLORS = {
    'TP': '#d3f9d8',  # green: metaphor in both
    'FP': '#ffe3e3',  # red: metaphor only in prediction
    'FN': '#fff3bf',  # yellow: metaphor only in ground truth
}
FONT = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto"

def tokenize_with_metaphor(xml_text: str, tag_name: str = "Metaphor") -> list[Token]:
    y_bin, y_bin_ref = xml_to_bin(xml_text, tag_name)
    return [Token(text=tok, is_metaphor=bool(flag)) for tok, flag in zip(y_bin_ref, y_bin)]


def tokens_to_html(tokens: list[Token],
                   metaphor_color: str = "#fff3bf",
                   text_color: str = "#000") -> str:
    spans: list[str] = []
    for t in tokens:
        style = f"background:{metaphor_color}; padding:2px 3px; border-radius:4px;" if t.is_metaphor else ""
        title = "Metaphor" if t.is_metaphor else "Literal/Other"
        spans.append(f"<span title='{title}' style='{style} color:{text_color};'>{escape(t.text)}</span>")
    return f"<div style='line-height:1.8; font-family:{FONT};'>" + " ".join(spans) + "</div>"

def align_for_display(pred_toks: list[Token], gt_toks: list[Token]) -> list[tuple[Token | None, Token | None]]:
    """Same alignment do_praf computes: ground truth first, prediction second.
    Returns (pred_token, gt_token) pairs; None marks a gap."""
    pairs = needleman_wunsch_indices([t.text for t in gt_toks], [t.text for t in pred_toks])
    return [
        (pred_toks[j] if j is not None else None,
         gt_toks[i] if i is not None else None)
        for i, j in pairs
    ]

def classify_pair(pred: Token | None, gt: Token | None) -> str:
    """TP/FP/FN/TN using the flags of each aligned position, a gap counting as 0 (as in do_praf)."""
    p = pred.is_metaphor if pred is not None else False
    g = gt.is_metaphor if gt is not None else False
    if p and g:
        return 'TP'
    if p:
        return 'FP'
    if g:
        return 'FN'
    return 'TN'

def count_categories(aligned: list[tuple[Token | None, Token | None]]) -> dict[str, int]:
    counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    for pred, gt in aligned:
        counts[classify_pair(pred, gt)] += 1
    return counts

def _cell(tok: Token | None, side: str, cat: str, differs: bool) -> str:
    background = f"background:{COLORS[cat]};" if cat in COLORS else ""
    border = "border:1px dashed #868e96;" if differs else "border:1px solid transparent;"
    if tok is None:
        title, text = f"{side}: (gap)", "—"
    else:
        title = f"{side}: {'Metaphor' if tok.is_metaphor else 'Not Metaphor'}"
        text = escape(tok.text)
    return (f"<span title='{title}' style='padding:2px 4px; border-radius:4px; {background} {border} "
            f"margin:1px; display:inline-block'>{text}</span>")

def aligned_html(pred_toks: list[Token], gt_toks: list[Token]) -> str:
    aligned = align_for_display(pred_toks, gt_toks)

    pred_cells: list[str] = []
    gt_cells: list[str] = []
    for pred, gt in aligned:
        cat = classify_pair(pred, gt)
        differs = pred is None or gt is None or pred.text != gt.text
        pred_cells.append(_cell(pred, "Pred", cat, differs))
        gt_cells.append(_cell(gt, "GT", cat, differs))

    counts = count_categories(aligned)
    legend = ("<div style='margin-bottom:8px'><strong>Legend:</strong> " +
              "".join(f"&nbsp; <span style='background:{COLORS[c]};padding:2px 6px;border-radius:4px'>{c}</span>"
                      for c in COLORS) +
              "&nbsp; <span style='border:1px dashed #868e96;padding:2px 6px;border-radius:4px'>tokens differ / gap</span>"
              f"&nbsp; &nbsp; TP={counts['TP']} FP={counts['FP']} FN={counts['FN']} TN={counts['TN']}</div>")

    return (f"<div style='font-family:{FONT}'>" + legend +
            "<div style='margin:6px 0'><strong>Predicted (aligned):</strong><br>" + " ".join(pred_cells) + "</div>"
            "<div style='margin:6px 0'><strong>Ground truth (aligned):</strong><br>" + " ".join(gt_cells) + "</div>"
            "</div>")

def visualize(predicted_text: str, ground_truth_text: str, tag_name: str = "Metaphor") -> str:
    pred_tokens = tokenize_with_metaphor(predicted_text, tag_name)
    gt_tokens = tokenize_with_metaphor(ground_truth_text, tag_name)
    return (
        "<h3>(1) Metaphor highlighting only</h3>"
        "<div style='display:flex; gap:24px; flex-wrap:wrap'>"
        f"<div><div style='font-weight:600; margin-bottom:6px'>Predicted</div>{tokens_to_html(pred_tokens, '#fff3bf')}</div>"
        f"<div><div style='font-weight:600; margin-bottom:6px'>Ground Truth</div>{tokens_to_html(gt_tokens, '#cfe8ff')}</div>"
        "</div>"
        "<h3>(2) Alignment with TP / FP / FN</h3>" + aligned_html(pred_tokens, gt_tokens)
    )

def save_html(body: str, path: str | Path) -> None:
    Path(path).write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{body}</body></html>",
                          encoding="utf-8")