from src.evaluation import xml_to_bin,needleman_wunsch_tokens
from pydantic import BaseModel
import re

class Token (BaseModel):
    text: str
    is_metaphor: bool

# Colors for categories
COLORS = {
    'TP': '#d3f9d8',       # green-ish
    'FP': '#ffe3e3',       # red-ish
    'FN': '#fff3bf',       # yellow-ish
    'MISMATCH': '#e9ecef', # gray
}

def tokenize_with_metaphor(xml_text: str, tag_name: str = "Metaphor") -> list[Token]:
    y_bin, y_bin_ref = xml_to_bin(xml_text, tag_name)
    return [Token(text=tok, is_metaphor=bool(flag)) for tok, flag in zip(y_bin_ref, y_bin)]

def needs_space(prev: str | None, curr: str | None) -> bool:
    """Basic English-ish spacing: no space before punctuation, add space between words."""
    if prev is None or curr is None:
        return False
    if re.fullmatch(r"[^\w\s]", curr):  # current is punctuation
        return False
    if re.fullmatch(r"[^\w\s]", prev):  # previous is punctuation
        return True
    return True

def tokens_to_html(tokens: list[Token],
                   metaphor_color: str = "#fff3bf",
                   text_color: str = "#000") -> str:
    html = ["<div style='line-height:1.8; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;'>"]
    prev = None
    for t in tokens:
        if needs_space(prev, t.text):
            html.append(" ")
        style = f"background:{metaphor_color}; padding:2px 3px; border-radius:4px;" if t.is_metaphor else ""
        title = "Metaphor" if t.is_metaphor else "Literal/Other"
        html.append(f"<span title='{title}' style='{style} color:{text_color};'>{t.text}</span>")
        prev = t.text
    html.append("</div>")
    return ''.join(html)

def align_tokens(pred_toks: list[str], gt_toks: list[str]) -> list[tuple[str | None, str | None]]:
    return needleman_wunsch_tokens(pred_toks, gt_toks)

def aligned_html(pred_toks: list[Token], gt_toks: list[Token]) -> str:
    pred_words = [t.text for t in pred_toks]
    gt_words = [t.text for t in gt_toks]
    alignment = align_tokens(pred_words, gt_words)

    # Map words -> metaphor flags in order (with counters for duplicates)
    from collections import defaultdict, deque
    pred_queue = defaultdict(deque)
    gt_queue = defaultdict(deque)
    for idx, t in enumerate(pred_toks):
        pred_queue[t.text].append(t.is_metaphor)
    for idx, t in enumerate(gt_toks):
        gt_queue[t.text].append(t.is_metaphor)

    rows = ["<div style='font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto'>"]
    rows.append("<div style='margin-bottom:8px'><strong>Legend:</strong> " +
                "&nbsp; <span style='background:%s;padding:2px 6px;border-radius:4px'>TP</span>" % COLORS['TP'] +
                "&nbsp; <span style='background:%s;padding:2px 6px;border-radius:4px'>FP</span>" % COLORS['FP'] +
                "&nbsp; <span style='background:%s;padding:2px 6px;border-radius:4px'>FN</span>" % COLORS['FN'] +
                "&nbsp; <span style='background:%s;padding:2px 6px;border-radius:4px'>Mismatch</span>" % COLORS['MISMATCH'] +
                "</div>")

    # Build two aligned rows
    pred_cells = []
    gt_cells = []
    for a, b in alignment:
        a_is_meta = None
        b_is_meta = None
        if a is not None:
            a_is_meta = pred_queue[a].popleft() if pred_queue[a] else False
        if b is not None:
            b_is_meta = gt_queue[b].popleft() if gt_queue[b] else False

        if a is None or b is None or a != b:
            cat = 'MISMATCH'
        else:
            if a_is_meta and b_is_meta:
                cat = 'TP'
            elif a_is_meta and not b_is_meta:
                cat = 'FP'
            elif (not a_is_meta) and b_is_meta:
                cat = 'FN'
            else:
                cat = None  # matched non-metaphor token

        style = f"background:{COLORS[cat]};" if cat else ""
        title_pred = f"Pred: {'Metaphor' if a_is_meta else 'Not Metaphor'}" if a is not None else "Pred: (gap)"
        title_gt   = f"GT: {'Metaphor' if b_is_meta else 'Not Metaphor'}" if b is not None else "GT: (gap)"

        pred_tok_html = a if a is not None else "—"
        gt_tok_html   = b if b is not None else "—"

        pred_cells.append(f"<span title='{title_pred}' style='padding:2px 4px; border-radius:4px; {style} margin:1px; display:inline-block'>{pred_tok_html}</span>")
        gt_cells.append(  f"<span title='{title_gt}'   style='padding:2px 4px; border-radius:4px; {style} margin:1px; display:inline-block'>{gt_tok_html}</span>")

    rows.append("<div style='margin:6px 0'><strong>Predicted (aligned):</strong><br>" + ' '.join(pred_cells) + "</div>")
    rows.append("<div style='margin:6px 0'><strong>Ground truth (aligned):</strong><br>" + ' '.join(gt_cells) + "</div>")
    rows.append("</div>")
    return ''.join(rows)