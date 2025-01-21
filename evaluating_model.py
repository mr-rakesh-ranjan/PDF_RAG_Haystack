
import fitz # PyMuPDF
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.evaluators import AnswerExactMatchEvaluator
from haystack.components.evaluators import SASEvaluator
from dotenv import load_dotenv, find_dotenv
import os


# Function to read PDF content 
def read_pdf(file_path): 
    pdf_document = fitz.open(file_path) 
    content = "" 
    for page in pdf_document: 
        content += page.get_text() 
    return content 

# Read custom PDF file 
pdf_content = read_pdf("./input_file.pdf") 

# Split the content into documents (Here, splitting by paragraphs for simplicity) 
documents = [Document(content=paragraph) for paragraph in pdf_content.split("\n\n")] 

# Write documents to InMemoryDocumentStore 
document_store = InMemoryDocumentStore() 
document_store.write_documents(documents)


# Build a RAG pipeline
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

# API_KEYS for OpenAI
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")


retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(api_key=Secret.from_token(api_key))

# for evaluating the model
ev_pipieline = Pipeline()
em_evaluator = AnswerExactMatchEvaluator()
sas_evaluator = SASEvaluator()
ev_pipieline.add_component("em_evaluator", em_evaluator)
ev_pipieline.add_component("sas_evaluator", sas_evaluator)
from questions_test import answers_list as ground_truth_answers, predicted_answers as predicted_answers
result = ev_pipieline.run(
    {
        "em_evaluator" : {"ground_truth_answers": ground_truth_answers, "predicted_answers": predicted_answers},
        "sas_evaluator" : {"ground_truth_answers": ground_truth_answers, "predicted_answers": predicted_answers},
    }
)

for evaluator in result:
    print(result[evaluator]["individual_scores"])

for evaluator in result:
    print(result[evaluator]["score"])

