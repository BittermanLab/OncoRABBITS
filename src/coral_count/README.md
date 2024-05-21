- You can download Coral dataset from Physionet at https://physionet.org/content/curated-oncology-reports/1.0/
- Download the datset and place the unzipped folder within coral_count

`batch_count.py` will process the unannotated dataset csv files within `src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data`

batch request files will be output in jsonl format to `src/coral_count/coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0/coral/unannotated/data/request`

These can be uploaded to the openai batch platform at https://platform.openai.com/