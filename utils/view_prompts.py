import html
from pathlib import Path
import pandas as pd

df = pd.read_csv("resources/prompts.csv").sort_values(["pid", "cid"])
colors = {"system": "#fff3cd", "user": "#e7f1ff", "assistant": "#e6f4ea"}
parts = ["<meta charset='utf-8'><style>body{font-family:sans-serif;max-width:900px;margin:2em auto}"
         ".m{padding:8px 12px;margin:6px 0;border-radius:6px;white-space:pre-wrap}"
         ".r{font-size:11px;font-weight:bold;text-transform:uppercase;color:#555}</style>"]
for (pid, name), group in df.groupby(["pid", "name"], sort=False):
    parts.append(f"<h2>{pid} - {html.escape(str(name))}</h2>")
    for _, row in group.iterrows():
        parts.append(f"<div class='m' style='background:{colors[row['role']]}'>"
                     f"<div class='r'>{row['role']}</div>{html.escape(row['content'])}</div>")
Path("outputs/prompts_view.html").write_text("".join(parts), encoding="utf-8")