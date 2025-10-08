# RAG for HQ Books

This repository contains resources and code for building a Retrieval-Augmented Generation (RAG) system tailored for handling high-quality (HQ) books. The main goal is to leverage advanced language models and retrieval techniques to answer questions and extract information from a collection of books.

## Features

- **Document Ingestion**: Tools and scripts for parsing and importing book content into a searchable format.
- **Retrieval System**: Efficient algorithms and methods to retrieve relevant book passages for a given query.
- **Generative QA**: Integration with language models to generate answers based on retrieved content.
- **Evaluation Tools**: Scripts for testing and evaluating the accuracy and relevance of generated answers.

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended: [poetry](https://python-poetry.org/) or [pipenv](https://pipenv.pypa.io/) for dependency management

### Installation

Clone the repository:

```bash
git clone https://github.com/sadiqabbas192/rag-for-hq-books.git
cd rag-for-hq-books
```

Install dependencies:

```bash
pip install -r requirements.txt
```
_or_  
```bash
poetry install
```

### Usage

1. **Prepare your book data**: Place your book files (TXT, PDF, etc.) in the designated `data/` directory.
2. **Ingest documents**: Run the document ingestion script to process your books into the retrieval system.
3. **Ask questions**: Use the main interface or API to query the system and get answers based on the book content.

Example:

```python
# Example usage (update with your main script)
from rag_for_hq_books import query

answer = query("What is the main theme of Book X?")
print(answer)
```

### Directory Structure

```
rag-for-hq-books/
├── data/               # Book files
├── ingestion/          # Scripts for data ingestion
├── retrieval/          # Retrieval model code
├── generation/         # Generative QA code
├── evaluation/         # Evaluation scripts and notebooks
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please open issues or pull requests with improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, open an issue or contact [sadiqabbas192](https://github.com/sadiqabbas192).
