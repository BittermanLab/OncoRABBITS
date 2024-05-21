## Coral
- Potential questions
  - Filter by those with medications-- *Given the note above,*
    - What class of drug is this 
    - What year was this approved
    - what is the chance of relapse/remission/cure
    - what is the chance of side effects in future
    - should we change the patients regime?


## GPT4o diffs
- check one to many concept mapping

## Jackson
- make an account on physionet
- download coral from physionet-> https://physionet.org/content/curated-oncology-reports/1.0/ -> put in coral_count folder 
- run `src/coral_count/batch_count.py` -> look at `combined_notes.csv` in the unannotated data folder (this is the unmodified notes in one file)
- run `src/coral_count/run_count.py`-> look at summary parquets in the counts folder
- run `src/coral_count/run_process.py` -> look at the datasets in the data/coral folder and compare to the original notes