from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack.components.builders.prompt_builder import PromptBuilder
from dotenv import load_dotenv, find_dotenv
import os

# API_KEYS for OpenAI
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")

pipe = Pipeline()
pipe.add_component("prompt_builder", PromptBuilder(template="{{ Questions }}?"))
pipe.add_component("llm", OpenAIGenerator(api_key=Secret.from_token(api_key)))
pipe.connect("prompt_builder", "llm")

res = pipe.run({"prompt_builder": {"country": "France"}})
# returns {"llm": {"replies": ['The official language of France is French.'] }}
print(res)
