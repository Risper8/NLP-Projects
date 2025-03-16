# Medical Note Extraction - H2O GenAI World NY

## Overview

This challenge focused on transforming medical notes into a structured JSON format, making the data more accessible and easier to analyze.

## Features

- **medical-extraction.ipynb**: A notebook used to train the T5 model for identifying and extracting key information from medical notes, which is then structured into a JSON format.
  
- **data_agent.ipynb**: A mini AI agent that uses the Gemini-1.5-Pro model to restructure the predicted data into a well-organized JSON format.

## Data

- **test_submission1 (3).csv**: A dataset containing the predicted outputs from the T5 model in a non-JSON format.

## Technologies Used

- **Transformers**: Used to manage the language models that generate responses from medical notes.
- **Vertex AI**: A platform for accessing Gemini models, which are used for data processing and transformation.

