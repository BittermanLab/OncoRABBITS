import os
from typing import List
from openai import OpenAI, AzureOpenAI

gpt_key = os.getenv("GPT4V_KEY")
openai_key = os.getenv("OPENAI_KEY")

SYS_PROMPT = 'You are an AI assistant answering multiple choice questions. Answer the questions using only the corresponding numbers for the answer.'

def get_chat_completion(
    user_prompt,
    system_prompt=SYS_PROMPT,
    engine="gpt-35-turbo-0613",
    service="chat",
    temperature=0,
    max_tokens=1,
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    full_response=False,
    stop=None,
):
    """
    Generates a chat completion using OpenAI's GPT model.

    Parameters:
    - user_prompt (str): The user's input prompt to the model.
    - system_prompt (str): The system's initial prompt setting the context for the model.
    - engine (str): The model you are using: [gpt-4, gpt4-32k, gpt-35-turbo-0613, 16k]
    - temperature (float): Controls randomness in the generation.
    - max_tokens (int): The maximum number of tokens to generate in the completion.
    - top_p (float): Nucleus sampling parameter controlling the size of the probability mass considered for token generation.
    - frequency_penalty (float): How much to penalize new tokens based on their frequency.
    - presence_penalty (float): How much to penalize new tokens based on their presence.
    - stop (list or None): Tokens at which to stop generating further tokens.

    Returns:
    - str: The generated completion text.
    """
    message_text = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if service == "azure":
        # print('using azure')
        client = AzureOpenAI(
            azure_endpoint="https://slt-openai-eastus2-service.openai.azure.com/",
            api_key=gpt_key,
            api_version="2024-02-15-preview",
        )
        completion = client.chat.completions.create(
            model=engine,
            messages=message_text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logprobs=True,
            top_logprobs=5,
        )
        if full_response:
            return completion
        else:
            return completion.choices[0].logprobs.content[0].top_logprobs

    else:
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=message_text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            logprobs=True,
            top_logprobs=5,
        )
        if full_response:
            return response
        else:
            return response.choices[0].message.content[0]


# Define a function to extract the top digit token from a list of TopLogprob objects
def extract_azure_top_digit(logprobs: List[dict]) -> str:
    """
    Extract the top digit token from a list of dictionaries representing TopLogprob objects.

    Args:
    - logprobs (List[dict]): A list of dictionaries, each representing a TopLogprob object with 'token' and 'logprob' keys.

    Returns:
    - str: The top digit token, or an empty string if no digit is found.
    """
    # Loop through each logprob object to find the first digit token
    for logprob in logprobs:
        token = logprob.token
        if token.isdigit():
            return token
    return ""  # Return an empty string if no digit is found


def extract_openai_top_digit(chat_completion):
    # Navigate through the object's structure to the list of tokens
    if not chat_completion.choices:
        return ""

    content_list = chat_completion.choices[0].logprobs.content

    # Find the first token that is a digit
    for token_logprob in content_list:
        if token_logprob.token.isdigit():
            return token_logprob.token

    return ""  # Return an empty string if no digit token is found
