from dotenv import load_dotenv
import os
from src.dataset_utils import load_dataset
from src.openai_utils import openai_initializer, prompt_openai_simple

load_dotenv()

dataset = load_dataset()
openai_key = os.getenv('OPENAI_API_KEY')
test_text = dataset.iloc[0]['plain']
print('Input text:')
print(test_text[:200] + '...' if len(test_text) > 200 else test_text)

if openai_key:
    openai_client = openai_initializer(openai_key)
    result_openai_prompt = prompt_openai_simple(openai_client, test_text)
    print(result_openai_prompt)