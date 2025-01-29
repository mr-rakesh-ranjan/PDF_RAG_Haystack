from ragas import evaluate
from datasets import Dataset, features
from questions_test import questions_list, answers_list, predicted_answers, retrieved_contexts

# Function to convert lists into dictionary
def list_into_dict(questions, answers, predicted_answers, retrieved_contexts):
    # Merging the lists into a list of dictionaries
    if len(questions) == len(answers) == len(predicted_answers):
        merged_list = [{'question': questions[i], 'ground_truth': answers[i], 'answer': predicted_answers[i],'contexts': retrieved_contexts[i]} for i in range(len(questions))]
        return merged_list
    else:
        print("All lists must have the same length")
        return None
    
# Convert lists into dictionary
# data = list_into_dict(questions_list, answers_list, predicted_answers, retrieved_contexts)
# # print(type(data), len(data))
data = [
    {
        "question": "What is the primary objective of the IIMA HR Policy Manual 2024?",
        "ground_truth": "What are the key steps in the recruitment process at IIMA for managerial positions?",
        "answer": "Who constitutes the Board of Governors at IIMA?",
        "contexts": "The primary objective of the IIMA HR Policy Manual 2024 is to provide a comprehensive framework for the management of human resources at IIMA. The manual outlines the policies, procedures, and guidelines that govern the recruitment, selection, training, development, performance evaluation, and compensation of employees at IIMA. The manual also establishes the roles and responsibilities of employees, supervisors, and the human resources department in the management of human resources at IIMA."
    }
]
datasets = Dataset.from_list(data)

result = evaluate(dataset=datasets)
print(result)