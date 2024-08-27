# Source Code Structure

The source code for this project is organized into several subdirectories within the `/src` directory, each with a specific purpose:

- `/src/hemonc_processing`: This directory contains the code that processes the R files from HemOnc and combines them into a single DataFrame. It also contains the code to process the PrimeKG data and create the mcqs for drug contraindications.

- `/src/request_generator`: This directory contains the code that processes the combined DataFrame and generates the batch requests for the OpenAI API.

- `/src/evaluation`: This directory contains the code that processes the batch responses from the API and evaluates the generated questions.

- `/src/irAE`: This directory contains the code for loading, batching, and evaluating the irAE detection cases and symptoms then create the mcqs for irAE detection.
  