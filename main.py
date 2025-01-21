
import fitz # PyMuPDF
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
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

# for rag pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")


predicted_answers =[]
from questions_test import questions_list as questions

for question in questions:
    print(question, "\n\n Answer:")
    result = rag_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    print(result['llm']['replies'])
    predicted_answers.append(result['llm']['replies'])

print("Predicted answer> \n\n",predicted_answers)


# while True:
#     query=input("\nEnter a query: ")
#     if query == "exit":
#         break
#     if query.strip() == "":
#         continue

#     results = rag_pipeline.run(
#         {
#             "retriever": {"query": query},
#             "prompt_builder": {"question": query},
#         }
#     )

#     #Print the result
#     print("\n\n> Question:")
#     print (query)
#     # print (f"\n> Answer (took {round (end - start, 2)} s.):") 
#     print("Answer - \n>",results["llm"]["replies"])
