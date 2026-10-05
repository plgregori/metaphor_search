from openai import OpenAI
from openai.types.file_object import FileObject
from pandas import DataFrame
import json
from transformers import (AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase,
                          PreTrainedModel, BatchEncoding, TrainingArguments, Trainer)
from transformers.trainer_utils import TrainOutput
import torch
from typing import cast
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from datasets import Dataset
import re

'''This part of the file is taken from the FineTuning_OpenAI notebook of the old scripts folder.
The bugs and dead code were fixed. The code in openai_utils.py is better and more complete, but this code is probably closer to what was actually used in the paper.'''

def prepare_data(data_df: DataFrame,
                 seed: int = 1,
                 train_ratio: float = 0.8) -> tuple[DataFrame, DataFrame]:
    '''This function is used both in the OpenAI and in the Transformers implementation'''
    train_df=data_df.sample(frac=train_ratio,random_state=seed)
    test_df=data_df.drop(index=train_df.index)
    return train_df, test_df

def prepare_chat_openai(train_df: DataFrame,
                 client: OpenAI,
                 train_ratio: float = 0.8,
                 seed: int = 1) -> FileObject:

    jsonl_path = f"corpus/ft_tr{train_ratio}_s{seed}.jsonl"

    user_msg_0="Can you please identify and tag the metaphors in the following text?"

    json_lines: list[str] = []
    for idx in range(0,train_df.shape[0]):
        
        raw_text = train_df.iloc[idx]["plain"].replace("\n"," ")
        text = train_df.iloc[idx]["metaphor_tagged_text"].replace('\\n', ' ')
        this_chat: dict[str, list[dict[str, str]]] = {
            "messages":[
                {"role":"user","content":user_msg_0+"\n"+raw_text},
                {"role":"assistant","content":text},
            ]
        }
        json_str = json.dumps(this_chat)
        json_lines.append(json_str)

    with open(jsonl_path,"w",-1) as f:
        f.write("\n".join(json_lines))

    fileinfo = client.files.create(file=open(jsonl_path, "rb"),purpose="fine-tune")
    print(fileinfo.id)

    return fileinfo

'''This part of the file is taken from the FineTuning_Transformers notebook of the old scripts folder.'''

def load_model(model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
               tokenizer_name: str = "meta-llama/Llama-3.2-1B-Instruct") -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer=cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizer_name)) # pyright: ignore[reportUnknownMemberType]
    model=cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")) # pyright: ignore[reportUnknownMemberType]

    tokenizer.pad_token = cast(str, tokenizer.eos_token)
    model.config.pad_token_id = cast(int, tokenizer.eos_token_id)
    return tokenizer, model

def prepare_chat_transformers(train_df: DataFrame) -> list[list[dict[str, str]]]:
    user_msg_0="Can you please identify and tag the metaphors in the following text? "

    messages_lists: list[list[dict[str, str]]] = []
    for idx in range(0,train_df.shape[0]):
        text = train_df.iloc[idx]["metaphor_tagged_text"].replace("\n"," ")
        raw_text = train_df.iloc[idx]["plain"].replace("\n"," ")
        messages: list[dict[str, str]] = [
                {"role":"user","content":user_msg_0+"\n"+raw_text},
                {"role":"assistant","content":text},
            ]
        messages_lists.append(messages)
    return messages_lists


def tokenize_chat(messages_lists: list[list[dict[str, str]]],
                  tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
    templated = cast(
        list[str],
        tokenizer.apply_chat_template(messages_lists, tokenize=False),  # pyright: ignore[reportUnknownMemberType]
    )
    encoded = tokenizer(templated, padding=True, truncation=True)
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

def setup_fine_tuning(model: PreTrainedModel, encoded_dataset: BatchEncoding) -> tuple[Trainer, PeftModel]:
    peft_config = LoraConfig(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, peft_config)
    if not isinstance(peft_model, PeftModel):
        raise TypeError("peft_model is expected to be of type PeftModel: something went wrong!")

    training_args = TrainingArguments(
        output_dir="chatbot",
        per_device_train_batch_size=1,
        eval_strategy="no",
        report_to="none",
        fp16=True,
    )

    train_dataset = Dataset.from_dict(dict(encoded_dataset))  # pyright: ignore[reportUnknownMemberType]

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset)
    return trainer, peft_model

def train_and_save(trainer: Trainer, output_dir: str = "ft_local_llama1b") -> TrainOutput:
    train_result = trainer.train() # pyright: ignore[reportUnknownMemberType]
    trainer.save_model(output_dir)
    return train_result

def infer(test_df: DataFrame, idx: int, tokenizer: PreTrainedTokenizerBase, model: PeftModel) -> str:
    raw_text = test_df.iloc[idx]["plain"].replace("\n"," ")
    user_msg_0 = "Can you please identify and tag the metaphors in the following text?"

    messages: list[dict[str, str]] = [
            {"role":"user","content":user_msg_0 + "\n" + raw_text},
        ]

    templated = cast(
            str,
            tokenizer.apply_chat_template(messages, tokenize = False),  # pyright: ignore[reportUnknownMemberType]
        )
    
    encoded = tokenizer(templated, return_tensors = "pt", truncation = True)

    outputs = model.generate(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        input_ids=encoded["input_ids"].to("cuda"),
        attention_mask=encoded["attention_mask"].to("cuda"),
    )

    out_text = cast(str, tokenizer.decode(outputs[0])) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]

    pttn="<\\|start_header_id\\|>assistant<\\|end_header_id\\|>(.*)"

    res = re.search(pttn,out_text, flags = re.DOTALL)
    if(res==None):
        rs = ""
    else:
        rs = res.group(1)

    return rs