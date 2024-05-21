## Coral
- create keywords df at `src/hemonc_processing/process_data.py`
- use b4b code for `src/coral_count/modify_coral.py` 
  - get count --> `coral_counts.csv` 
  - Replacement script--> `brand_coral.csv` and `generic_coral.csv`
- Potential questions
  - Filter by those with medications-- *Given the note above,*
    - What class of drug is this 
    - What year was this approved
    - what is the chance of relapse/remission/cure
    - what is the chance of side effects in future
    - should we change the patients regime?


## GPT4o diffs
- check one to many concept mapping
- Check model, infeffective, token count, temps
- Create eval script for each debug version
- Run on all 


## Jackson
- make an account on physionet
- download coral from physionet-> https://physionet.org/content/curated-oncology-reports/1.0/ -> put in coral_count folder as readme
- run `src/coral_count/batch_count.py` -> look at `combined_notes.csv` (this is the unmodified notes in one file)
- run `src/coral_count/modify_coral.py`-> look at the following:
  - `coral_counts.csv` -> this is the number of replacements in the regex-- look for words that should be included in replacement (words that are likely not drugs)
  - `brand_coral.csv` --> this is the `combined_notes.csv` with all generic drugs swapped to a brand equivalent-- check that we have replaced these words correctly 
  - `generic_coral.csv` --> this is the `combined_notes.csv` with all brand drugs swapped to a generic equivalent-- check that we have replaced these words correctly
