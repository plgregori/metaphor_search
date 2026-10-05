import pandas as pd
from src.visualization import visualize, save_html

ex = pd.read_csv("results/Example.csv")
gold = pd.read_csv("corpus/metaphor_dataset.csv").set_index("textid")

row = ex[ex.prompt_strategy == "Zero shot"].iloc[0]
save_html(visualize(row["answer"], gold.loc[row["textid"], "metaphor_tagged_text"]), "outputs/example.html")