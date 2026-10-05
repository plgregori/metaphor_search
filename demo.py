from src.dataset_utils import load_dataset, dataset_details, load_dataset_for_rag, split_dataset_for_rag
from src.openai_utils import (openai_initializer, prompt_openai_simple, prepare_finetuning_data_openai,
                              prepare_rag_context, prompt_openai_rag, run_all_prompts_on_dataset,
                              summarise_experiment_costs, load_and_check_prompt_data_openai,
                              extract_system_instruction, save_training_jsonl, prepare_finetuning_data_openai_bis,
                              run_fine_tuned_model_on_dataset, submit_finetuning_jsonl)
from src.ollama_utils import prompt_ollama
from src.rag_utils import get_embeddings_batch, normalise_and_save_embeddings, run_rag_on_dataset
from ollama import Client
from src.transformers_utils import prepare_finetuning_data_transformers
from src.evaluation import (parse_metaphor_tags, evaluate_all_samples, evaluate_predictions, identify_no_metaphors,
                            summarize_scores)
from src.prompt_engineering import dataset_preparation, load_prompt_strategies
from dotenv import load_dotenv
import os
from pathlib import Path

CALL_OPENAI = False
GENERIC_DEMO = False
PROMPT_ENGINEERING = True
FINE_TUNING = True
RAG = True

TOP_K = 3

load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')
ollama_url = os.getenv('OLLAMA_URL')

openai_client = openai_initializer(str(openai_key))

Path("outputs").mkdir(parents=True, exist_ok=True)

# Generic demo
"""This is mostly taken from the Comprehensive_Metaphor_Identification_Workflow notebook, which is just a demo for illustration purposes."""
if GENERIC_DEMO:
    # Load dataset
    dataset = load_dataset('corpus/metaphor_dataset.csv')

    # Dataset facts
    dataset_details(dataset)
    test_text = dataset.iloc[0]['plain']
    print(f"This is the test text:\n\n{test_text}")

    # Prepare fine-tuning data
    train_df_ft, test_df_ft, jsonl_path = prepare_finetuning_data_openai(dataset)
    print(f'Fine-tuning data prepared:')
    print(f'  Train set: {len(train_df_ft)} samples')
    print(f'  Test set: {len(test_df_ft)} samples')
    print(f'  JSONL file: {jsonl_path}')

    # OpenAI Section
    if CALL_OPENAI:
        # Prompt with OpenAI        
        print('Input text:')
        print(test_text[:200] + '...' if len(test_text) > 200 else test_text)
        print('\nProcessing with OpenAI...')
        result_openai_prompt = prompt_openai_simple(openai_client, test_text)
        print('\nOutput:')
        print(result_openai_prompt)
        # Fine-tuning with OpenAI
        with open(jsonl_path, 'rb') as f:
            fileinfo = openai_client.files.create(file=f, purpose='fine-tune')     

            ft_job = openai_client.fine_tuning.jobs.create(
                                training_file=fileinfo.id,
                                model='gpt-4.1-mini-2025-04-14',
                                suffix='metaphor_ft'
                                    )
            print(f'Fine-tuning job created: {ft_job.id}')
            ft_model_id = None  # Will be available after training completes

        rag_context = prepare_rag_context()
        print('RAG context prepared')
        print(f'Context length: {len(rag_context)} characters')

        # Test RAG on first example
        test_text_rag = dataset.iloc[0]['plain']
        print('Testing RAG approach...')
        result_openai_rag = prompt_openai_rag(openai_client, test_text_rag, rag_context)
        if result_openai_rag:
            print('RAG Result:' , result_openai_rag[:300])

    # Prompt with Ollama
    ollama_client = Client(host = ollama_url)
    print('Testing Ollama (llama3.2:1b)...')
    result_ollama = prompt_ollama(ollama_client, test_text, model='llama3.2:1b')
    print('Result:', result_ollama[:300] if result_ollama else 'Failed')

    # Prepare dataset for transformer fine-tuning
    train_df_transformers, test_df_transformers = prepare_finetuning_data_transformers(dataset)
    print(f'Data prepared for Transformers fine-tuning')
    print(f'Train: {len(train_df_transformers)}, Test: {len(test_df_transformers)}')

    # Evaluation
    example_true = dataset.iloc[0]['metaphor_tagged_text']
    example_pred = example_true # For demo, we'll use the same as prediction (perfect match)
    print('\nEvaluation Example:')
    print(f'True tags found: {len(parse_metaphor_tags(example_true))}')
    print(f'Pred tags found: {len(parse_metaphor_tags(example_pred))}')

    print('Running comprehensive evaluation on all samples...')
    evaluation_results = evaluate_all_samples(dataset)
    if evaluation_results:
        print('\n' + '='*60)
        print('COMPREHENSIVE EVALUATION RESULTS')
        print('='*60)
        print(f'Number of samples evaluated: {evaluation_results.num_samples}')
        print(f'\nAverage Precision: {evaluation_results.precision:.4f}')
        print(f'Average Recall:    {evaluation_results.recall:.4f}')
        print(f'Average Accuracy:  {evaluation_results.accuracy:.4f}')
        print(f'Average F1 Score:  {evaluation_results.f1:.4f}')
        print('='*60)


# Prompt engineering
if PROMPT_ENGINEERING and CALL_OPENAI:
    dataset_df, prompts_df = dataset_preparation(dataset_path = "corpus/metaphor_dataset.csv", prompts_path= "resources/prompts.csv")
    print(dataset_df.head())
    print(prompts_df.head(10))

    prompt_messages_by_pid, prompt_names_by_pid = load_prompt_strategies(prompts_df)

    print("Loaded prompt ids:", list(prompt_messages_by_pid.keys()))
    print("Example prompt name:", prompt_names_by_pid[list(prompt_names_by_pid.keys())[0]])

    results_df = run_all_prompts_on_dataset(
        dataset=dataset_df,
        prompt_messages_by_pid=prompt_messages_by_pid,
        prompt_names_by_pid=prompt_names_by_pid,
        client=openai_client,
        model_name="gpt-5.4-mini",
        num_repeats=2,
        sleep_seconds=0.2,
    )

    print(f"Total result rows: {len(results_df)}")
    print(results_df.head())

    cost_summary_df = summarise_experiment_costs(results_df)

    outputs_csv_path = "outputs/llm_outputs_by_prompt_PE.csv"
    cost_summary_csv_path = "outputs/experiment_cost_summary_PE.csv"

    results_df.to_csv(outputs_csv_path, index=False)
    cost_summary_df.to_csv(cost_summary_csv_path, index=False)

    print(f"Saved model outputs to: {outputs_csv_path}")
    print(f"Saved cost summary to: {cost_summary_csv_path}")

    #Evaluation of the prompt-engineering
    row_level_metrics_df = evaluate_predictions(results_df)
    print(f"Scored rows: {len(row_level_metrics_df)}")
    print(row_level_metrics_df.head())

    #Checking rows with no metaphors
    missing_metaphor_csv_path = "outputs/lines_without_metaphor_tags_PE.csv"
    missing_metaphor_summary_df = identify_no_metaphors(results_df, missing_metaphor_csv_path)
    print(f"Saved rows without metaphor tags to: {missing_metaphor_csv_path}")
    print(f"Rows with no <Metaphor>...</Metaphor> tag pair: {len(missing_metaphor_summary_df)}")
    print(missing_metaphor_summary_df.head(20))

    #Summarizing the score metrics
    prompt_level_metrics_df = summarize_scores(row_level_metrics_df)

    #Saving the evaluation files to CSV
    row_metrics_csv_path = Path("outputs/row_level_metrics_PE.csv")
    prompt_metrics_csv_path = Path("outputs/prompt_level_metrics_PE.csv")

    row_level_metrics_df.to_csv(row_metrics_csv_path, index=False)
    prompt_level_metrics_df.to_csv(prompt_metrics_csv_path, index=False)

    print(f"Saved row-level metrics to: {row_metrics_csv_path.resolve()}")
    print(f"Saved prompt-level metrics to: {prompt_metrics_csv_path.resolve()}")

    #Inspecting a few examples
    example_view_df = (
        results_df.merge(
            row_level_metrics_df,
            on=["prompt_id", "prompt_name", "repeat_number", "dataset_index", "textid", "model_name"],
            how="left",
            suffixes=("", "_metric"),
        )
    )

    print(example_view_df[[
        "prompt_id",
        "prompt_name",
        "repeat_number",
        "textid",
        "plain",
        "gold_metaphor_tagged_text",
        "llm_output",
        "precision",
        "recall",
        "accuracy",
        "f1",
        "estimated_cost_usd",
    ]].head(10))

    """What the output files mean

    After the notebook finishes, you should have these files in the `outputs` folder:

    - `llm_outputs_by_prompt.csv`  
    One row per prompt x text x repeat. Includes the raw model output, token usage, cost estimate and any API error message.

    - `experiment_cost_summary.csv`  
    A compact summary of the number of requests, tokens used and estimated total cost by model.

    - `lines_without_metaphor_tags.csv`  
    A list of rows where the model output did not contain any `<Metaphor>...</Metaphor>` tag pair.

    - `row_level_metrics.csv`  
    One row per successfully evaluated prompt x text x repeat with precision, recall, accuracy and F1.

    - `prompt_level_metrics.csv`  
    One summary row per prompt strategy, averaging the evaluation metrics across all scored rows and summing the estimated cost."""

if FINE_TUNING and CALL_OPENAI:
    dataset_path = "corpus/metaphor_dataset.csv"
    prompts_path = "resources/prompts.csv"

    # First we take the two CSV files and turn them into pandas DataFrames, and check they have the correct shape
    dataset_df, prompts_df = load_and_check_prompt_data_openai(dataset_path, prompts_path)
    print(dataset_df.head())
    print(prompts_df.head(10))

    # Now we extract the system instruction from the prompts Dataframe
    system_instruction = extract_system_instruction(prompts_df)
    print("System instruction (first 200 chars):")
    print(system_instruction[:200])

    # New we split the dataset Dataframe in train and test sets
    train_df, test_df = prepare_finetuning_data_openai_bis(dataset_df)

    # We turn the train Dataframe into a jsonl of chat conversations
    save_training_jsonl(train_df, system_instruction, "outputs/fine_tune_train_FT.jsonl")

    # We submit the fine-tuning job
    fine_tuned_model_name = submit_finetuning_jsonl(openai_client, "outputs/fine_tune_train_FT.jsonl")

    # We now run the fine-tuned model on the test dataset
    results_df = run_fine_tuned_model_on_dataset(
        dataset=test_df,
        system_instruction=system_instruction,
        client=openai_client,
        model_name=str(fine_tuned_model_name))

    print(f"Total result rows: {len(results_df)}")
    print(results_df.head())

    cost_summary_df = summarise_experiment_costs(results_df)

    print(cost_summary_df)
    results_path = Path("outputs/llm_outputs_fine_tuned_FT.csv")
    cost_summary_path = Path("outputs/experiment_cost_summary_FT.csv")
    results_df.to_csv(results_path, index=False)
    cost_summary_df.to_csv(cost_summary_path, index=False)

    print(f"Saved model outputs to:  {results_path.resolve()}")
    print(f"Saved cost summary to:   {cost_summary_path.resolve()}")

    row_level_metrics_df = evaluate_predictions(results_df)
    print(f"Scored rows: {len(row_level_metrics_df)}")
    print(row_level_metrics_df.head())

    #Checking rows with no metaphors
    missing_metaphor_csv_path = "outputs/lines_without_metaphor_tags_FT.csv"
    missing_metaphor_summary_df = identify_no_metaphors(results_df, missing_metaphor_csv_path)
    print(f"Saved rows without metaphor tags to: {missing_metaphor_csv_path}")
    print(f"Rows with no <Metaphor>...</Metaphor> tag pair: {len(missing_metaphor_summary_df)}")
    print(missing_metaphor_summary_df.head(20))

    #Summarizing the score metrics
    model_level_metrics_df = summarize_scores(row_level_metrics_df)

    row_metrics_csv_path   = Path("outputs/row_level_metrics_FT.csv")
    model_metrics_csv_path = Path("outputs/model_level_metrics_FT.csv")

    row_level_metrics_df.to_csv(row_metrics_csv_path, index=False)
    model_level_metrics_df.to_csv(model_metrics_csv_path, index=False)

    print(f"Saved row-level metrics to:   {row_metrics_csv_path.resolve()}")
    print(f"Saved model-level metrics to: {model_metrics_csv_path.resolve()}")

    #Inspecting a few example predictions
    example_view_df = (
    results_df.merge(
            row_level_metrics_df,
            on=["model_name", "repeat_number", "dataset_index", "textid"],
            how="left",
            suffixes=("", "_metric"),
        )
    )

    print(example_view_df[[
        "model_name",
        "repeat_number",
        "textid",
        "plain",
        "gold_metaphor_tagged_text",
        "llm_output",
        "precision",
        "recall",
        "accuracy",
        "f1",
        "estimated_cost_usd",
    ]].head(10))

    """What the output files mean

    After the notebook finishes, you should have these files in the `outputs` folder:

    - `fine_tune_train.jsonl` 
    The JSONL training file uploaded to the OpenAI Files API. Each line is one training example.

    - `fine_tuned_model_id.txt` 
    The ID of the fine-tuned model (e.g. `ft:gpt-5-mini:org:metaphor:XXXXXXXX`).   Paste this into `FINE_TUNED_MODEL_NAME` if you want to skip re-training and run inference only.

    - `llm_outputs_fine_tuned.csv` 
    One row per test text x repeat. Includes the raw model output, token usage, cost estimate   and any API error message.

    - `experiment_cost_summary.csv` 
    A compact summary of the number of inference requests, tokens used and estimated total cost by model.

    - `lines_without_metaphor_tags.csv` 
    A list of rows where the model output did not contain any `<Metaphor>...</Metaphor>` tag pair.

    - `row_level_metrics.csv` 
    One row per successfully evaluated test text x repeat with precision, recall, accuracy and F1.

    - `model_level_metrics.csv` 
    One summary row for the fine-tuned model, averaging the evaluation metrics across all   scored rows and summing the estimated inference cost."""

if RAG and CALL_OPENAI:
    dataset_df, prompts_df = load_dataset_for_rag()
    print(dataset_df.head())
    print(prompts_df.head(10))
    train_df, test_df = split_dataset_for_rag(dataset_df, top_k=TOP_K)
    prompt_messages_by_pid, prompt_names_by_pid = load_prompt_strategies(prompts_df)
    print("Loaded prompt ids:   ", list(prompt_messages_by_pid.keys()))
    print("Example prompt name:", prompt_names_by_pid[list(prompt_names_by_pid.keys())[0]])

    print(f"Embedding {len(train_df)} training texts with model 'text-embedding-3-small'...")
    train_embeddings = get_embeddings_batch(
        texts=train_df["plain"].tolist(),
        client=openai_client,
    )
    train_embeddings_normed = normalise_and_save_embeddings(train_embeddings)
    results_df = run_rag_on_dataset(
        dataset=test_df,
        train_pool=train_df,
        train_embeddings_normed=train_embeddings_normed,
        prompt_messages_by_pid=prompt_messages_by_pid,
        prompt_names_by_pid=prompt_names_by_pid,
        client=openai_client,
        top_k=TOP_K
    )

    print(f"Total result rows: {len(results_df)}")
    print(results_df.head())

    cost_summary_df = summarise_experiment_costs(results_df)
    print(cost_summary_df)

    # Save raw model outputs and cost summary to CSV
    outputs_csv_path = "outputs/llm_outputs_rag_RAG.csv"
    cost_summary_csv_path = "outputs/experiment_cost_summary_RAG.csv"
    results_df.to_csv(outputs_csv_path, index=False)
    cost_summary_df.to_csv(cost_summary_csv_path, index=False)

    print(f"Saved model outputs to: {outputs_csv_path}")
    print(f"Saved cost summary to:  {cost_summary_csv_path}")

    # Evaluation
    row_level_metrics_df = evaluate_predictions(results_df)
    print(f"Scored rows: {len(row_level_metrics_df)}")
    print(row_level_metrics_df.head())

    # Identify no metaphors

    missing_metaphor_summary_df = identify_no_metaphors(results_df, "outputs/lines_without_metaphor_tags_RAG.csv")

    # Metrics aggregation
    prompt_level_metrics_df = summarize_scores(row_level_metrics_df)

    # Save evaluation files to CSV
    row_metrics_csv_path    = "outputs/row_level_metrics_RAG.csv"
    prompt_metrics_csv_path = "outputs/prompt_level_metrics_RAG.csv"

    row_level_metrics_df.to_csv(row_metrics_csv_path, index=False)
    prompt_level_metrics_df.to_csv(prompt_metrics_csv_path, index=False)

    print(f"Saved row-level metrics to:    {row_metrics_csv_path}")
    print(f"Saved prompt-level metrics to: {prompt_metrics_csv_path}") 

    # Inspect a few example predictions
    example_view_df = (
        results_df.merge(
            row_level_metrics_df,
            on=["prompt_id", "prompt_name", "repeat_number", "dataset_index", "textid", "model_name"],
            how="left",
            suffixes=("", "_metric"),
        )
    )

    print(example_view_df[[
        "prompt_id",
        "prompt_name",
        "repeat_number",
        "textid",
        "retrieved_textids",
        "plain",
        "gold_metaphor_tagged_text",
        "llm_output",
        "precision",
        "recall",
        "accuracy",
        "f1",
        "estimated_cost_usd",
    ]].head(10))

    """What the output files mean:
    After the notebook finishes, you should have these files in the `outputs` folder:

    - `train_embeddings.npy` 
    The normalised embedding matrix for the retrieval pool.   Shape: `(n_train, embedding_dim)`. Reload with `np.load()` to skip re-embedding.

    - `llm_outputs_rag.csv` 
    One row per prompt x test text x repeat. Includes the IDs of retrieved examples,   the raw model output, token usage, cost estimate and any API error message.

    - `experiment_cost_summary.csv` 
    A compact summary of the number of chat completions requests, tokens used and   estimated total cost by model.

    - `lines_without_metaphor_tags.csv` 
    A list of rows where the model output did not contain any `<Metaphor>...</Metaphor>` tag pair.

    - `row_level_metrics.csv` 
    One row per successfully evaluated prompt x test text x repeat with precision,   recall, accuracy and F1.

    - `prompt_level_metrics.csv` 
    One summary row per prompt strategy, averaging the evaluation metrics across all   scored rows and summing the estimated cost."""

        
    