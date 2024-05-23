## Coral
- Potential questions
  - Filter by those with medications-- *Given the note above,*
  - **"Identify the pharmacological class of the prescribed drugs."**
  - "In what year did the FDA approve this medication?"
  - "Estimate the likelihood of relapse for this patient."
  - "What are the chances of remission for the patient given the treatment plan?"
  - "How likely is it that the patient will achieve a cure with this treatment?"
  - **"Provide a prognosis based on the current treatment plan."**
  - "Is there a need to modify the patient's current treatment regimen?"
  - **"Would you recommend any changes to the patient's medication plan?"**
  - **"Does the current treatment align with established clinical guidelines?"**
  - "What is the efficacy rate of the prescribed medication for this type of cancer?"
  - "Based on the patient's profile, is this the most suitable medication?"
  - **"How frequently should the patient be monitored while on this medication?"**

- Potential other work
  - Sentiment (kind of ablation study)
  - Add the demographic columns to the data and re run- does it change the results?


## GPT4o diffs
- check one to many concept mapping

## Jackson
- make an account on physionet
- download coral from physionet-> https://physionet.org/content/curated-oncology-reports/1.0/ -> put in coral_count folder 
- run `src/coral_count/batch_count.py` -> look at `combined_notes.csv` in the unannotated data folder (this is the unmodified notes in one file)
- run `src/coral_count/run_count.py`-> look at summary parquets in the counts folder
- run `src/coral_count/run_process.py` -> look at the datasets in the data/coral folder and compare to the original notes