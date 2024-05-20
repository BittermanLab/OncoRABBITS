# Project Structure

This project is organized into several folders, each with a specific purpose:

- `data`: This is the main folder for all data used in this project.

- `api_responses`: This folder contains the batch files returned by the GPT-4 OpenAI API. These responses are used for further processing and analysis.

- `drug_names`: This folder contains the combined DataFrame from HemOnc and the processing to `combined_df.csv`. This is the starting point for all question generation.

- `request`: This folder is used for data processing to convert the `combined_df.csv` to the format needed for the API batch requests.