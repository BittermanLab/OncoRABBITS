import pandas as pd
import os
import json
from typing import List, Dict, Any
from list_utils import combine_and_plot_list_preference, plot_detailed_attribute_counts
from general_knowledge_utils import plot_general_knowledge
from sentiment_utils import plot_sentiment


# Directory setup
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)

# Task details
tasks = {
    # "general_knowledge": "general_knowledge",
    "sentiment_question_about": "sentiment",
    "sentiment_question_patient": "sentiment",
    "sentiment_question_physician": "sentiment",
    "list_preference_prompt1": "list_preference",
    "list_preference_prompt2": "list_preference",
    # "coral_sentiment": "sentiment",
}

models = ["gpt-3.5-turbo-0125", "gpt-4o", "gpt-4-turbo"]

# Iterate through each task and model
for task_name, file_name in tasks.items():
    for model in models:
        # Check output directory exists for each model
        model_output_dir = os.path.join(output_dir, model)
        os.makedirs(model_output_dir, exist_ok=True)

        print(f"Processing task '{task_name}' for model '{model}'")

        if (
            task_name == "list_preference_prompt1"
            or task_name == "list_preference_prompt2"
        ):
            model_output_dir = os.path.join(output_dir, model, "list_preference")
            combine_and_plot_list_preference(model_output_dir, model)
            plot_detailed_attribute_counts(model_output_dir, model)

        elif (
            task_name == "sentiment_question_about"
            or task_name == "sentiment_question_patient"
            or task_name == "sentiment_question_physician"
            or task_name == "coral_sentiment"
        ):
            plot_sentiment(model_output_dir, task_name, model)
        elif task_name == "general_knowledge":
            plot_general_knowledge(model_output_dir, model)
        else:
            print("Error: Task not found")

        print(f"{task_name} processing completed for model '{model}'.")
        print("\n" * 5)
