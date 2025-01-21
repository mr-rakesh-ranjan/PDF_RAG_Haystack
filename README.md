# Haystack POC

This repository contains a proof of concept (POC) for using the Haystack framework to build and evaluate a Retrieval-Augmented Generation (RAG) pipeline. The project includes components for reading PDF content, storing documents, retrieving relevant documents, generating answers using an LLM, and evaluating the generated answers.

## Project Structure

```
.
├── .env
├── .gitignore
├── app.py
├── evaluating_model.py
├── main.py
├── questions_test.py
├── questions.py
├── replicate_POC.py
└── requirements.txt
```

## Files

- **app.py**: Contains a simple pipeline for generating answers using OpenAI's GPT model.
- **evaluating_model.py**: Builds and evaluates a RAG pipeline using the Haystack framework.
- **main.py**: Main script for building and running the RAG pipeline.
- **questions_test.py**: Contains a list of questions and their corresponding ground truth answers for evaluation.
- **questions.py**: Contains a list of questions and their corresponding ground truth answers and predicted answers.
- **replicate_POC.py**: Example script for generating images using the Replicate API.
- **requirements.txt**: Lists the dependencies required for the project.

## Setup

1. **Clone the repository**:
    ```sh
    git clone <https://github.com/mr-rakesh-ranjan/PDF_RAG_Haystack>
    cd <PDF_RAG_Haystack>
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a [`.env`](.env ) file in the root directory and add your API keys:
      ```
      OPENAI_API_KEY="your_openai_api_key"
      REPLICATE_API_TOKEN="your_replicate_api_token"
      ```

## Usage

### Running the RAG Pipeline

1. **Run the main script**:
    ```sh
    python main.py
    ```

2. **Run the evaluation script**:
    ```sh
    python evaluating_model.py
    ```

### Generating Images with Replicate API

1. **Run the replicate POC script**:
    ```sh
    python replicate_POC.py
    ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Haystack](https://github.com/deepset-ai/haystack) - An open-source framework for building search systems.
- [OpenAI](https://openai.com/) - For providing the GPT model used in this project.
- [Replicate](https://replicate.com/) - For providing the API used to generate images.

## Contact

For any questions or inquiries, please contact [Rakesh Ranjan Kumar](mailto:rakeshranjan.java1.8@gmail.com).
