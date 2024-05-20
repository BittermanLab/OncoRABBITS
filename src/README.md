# Source Code Structure

The source code for this project is organized into several subdirectories within the `/src` directory, each with a specific purpose:

- `/src/hemonc_processing`: This directory contains the code that processes the R files from HemOnc and combines them into a single DataFrame.

- `/src/request_generator`: This directory contains the code that processes the combined DataFrame and generates the batch requests for the GPT-4 OpenAI API.

- `/src/evaluation`: This directory contains the code that processes the batch responses from the API and evaluates the generated questions.