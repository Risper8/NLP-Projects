# Skin Disease Prediction NLP Model

## Overview

This model is designed to predict skin diseases and associated symptoms based on user input. It can also identify symptoms from user input and provide a confidence interval for the predicted disease. It's essentially similar to a RAG application, but focuses more on the model aspect.

## Notebooks

- **skin_disease.ipynb**: Contains code related to the skin disease prediction model.
- **skin_disease_chromadb.ipynb**: Contains code for integrating a vector database with the model, enabling storage of text embeddings from both the dataset and user inputs.

## Data

The data used for the model is stored in JSON and CSV files.

- **vectordb**: The database created for storing text and vector embeddings.

## Tools and / Technologies Used

- Python
- Sentence Transformers
- spaCy
- ChromaDB
