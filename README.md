<!-- exclude_docs -->
# OncoRABBITS: Oncology-focused Robust Assessment of Biomedical Benchmarks Involving drug Term Substitutions

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE.txt)
[![Arxiv](https://img.shields.io/badge/Arxiv-Coming%20Soon-red)]()
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-OncoRABBITS-green)]()

**OncoRABBITS** extends the [RABBITS](https://github.com/BittermanLab/RABBITS) project, focusing specifically on evaluating the robustness of language models in the oncology domain through the substitution of brand names for generic drugs and vice versa. This evaluation is crucial as medical prescription errors, particularly in oncology, can have severe consequences for patient outcomes.

![OncoRABBITS Plot](onco_rabbits_plot.png)

## Motivation

Building on the work of RABBITS, OncoRABBITS narrows the focus to oncology drugs, where precise terminology is critical for patient safety and treatment efficacy. By modifying oncology-specific benchmarks and replacing drug names, we aim to:

1. Brand-generic drug pair identification
2. Drug attribute association
3. Drug contraindication detection
4. Immune-related Adverse Event (irAE) detection in oncology

Our findings reveal performance variations when oncology drug names are swapped, suggesting potential issues in the model's ability to generalize in this critical medical domain.


## Setup

### Data Sources
- HemOnc database: Used to generate lists of brand-generic drug pairs across all indications and time periods.
- PrimeKG knowledge graph: Used to extract drug interactions for contraindication analysis.

### Models Evaluated
- GPT-3.5 Turbo
- GPT-4 Turbo
- GPT-4o

### Temperature Settings
Models were evaluated across three temperature settings: 0.0, 0.7, and 1.0.

### Evaluation Tasks

1. **Brand-Generic Pair Identification**
   - Multiple-choice question format with 5-fold resampling of incorrect answers
   - Single-token and regex-based performance evaluation

2. **Drug Attribute Association**
   - List preference method for attribute association
   - Attributes: safe, unsafe, effective, ineffective, have side effects, side effect free
   - Regex patterns used to count term frequency
   - Chatbot simulation for sentiment analysis (using a RoBERTa-based model)

3. **Drug Contraindication Detection**
   - PrimeKG-derived drug interaction triplets converted to 4-option MCQs
   - Regex-based evaluation of contraindicated drug detection

4. **irAE Detection in Oncology**
   - Curated list of immunotherapy regimens from HemOnc dataset
   - Comprehensive list of irAE symptoms based on literature review
   - Claude 3.5 Sonnet-generated EHR contexts in oncQA style
   - Evaluation of differential diagnoses and irAE likelihood ranking

## Project Structure

- `src/`: Source code for data processing and analysis
- `data/`: Input data files and processed datasets
- `results/`: Output files and analysis results
- `notebooks/`: Jupyter notebooks for exploratory data analysis and visualization

## Setup and Usage

### Environment Setup

1. Clone the repository:
   ```
   git clone https://github.com/BittermanLab/OncoRABBITS.git
   cd OncoRABBITS
   ```

2. Create a conda environment from the environment.yml file:
   ```
   conda env create -f environment.yml
   conda activate onco_rabbits
   ```

### API Key Configuration

To keep API keys secure:

1. Make sure `python-dotenv` is installed:
   ```
   pip install python-dotenv
   ```

2. Create a `.env` file in the project root with your API keys:
   ```plaintext
   OPENAI_API_KEY=your_key_here
   ```

3. In your Python code, load the environment variables:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   import os

   api_key = os.getenv('OPENAI_API_KEY')
   ```

4. Add `.env` to your `.gitignore` file to prevent accidental commits.

### Running the Analysis

1. Prepare the data so run the code in folder `src/hemonc_processing`

2. Generate the batch requests for the OpenAI API using the code in folder `src/request_generator` and `src/irAE`
   
3. Perform the evaluation using the code in folder `src/evaluation`
   
## Results

All plots and results can be found in the `results/` folder.


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- HemOnc database for providing drug information
- PrimeKG knowledge graph for drug interaction data
- OpenAI for access to GPT models
- Claude AI for assistance in generating EHR contexts
- The RABBITS team for their initial work

## Citing

@article{onco_rabbits2024,
title={OncoRABBITS: Oncology-focused Robust Assessment of Biomedical Benchmarks Involving drug Term Substitutions},
author={[Author List]},
journal={[Journal]},
year={2024},
note={[Publication Date]}
}

For the foundational RABBITS work, please also cite:

@article{gallifant2024rabbits,
title={Language Models are Surprisingly Fragile to Drug Names in Biomedical Benchmarks},
author={Gallifant, Jack and Chen, Shan and Moreira, Pedro and Munch, Nikolaj and Gao, Mingye and Pond, Jackson and Aerts, Hugo and Celi, Leo Anthony and Hartvigsen, Thomas and Bitterman, Danielle S.},
journal={arXiv preprint arXiv:2406.12066v1 [cs.CL]},
year={2024},
note={17 Jun 2024}
}

## Contact

For any questions or concerns, please open an issue in this repository or contact the project maintainers.