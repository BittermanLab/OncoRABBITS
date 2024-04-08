<!-- exclude_docs -->
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE.txt)

<!-- exclude_docs_end -->

<!-- exclude_docs -->
> **⚗️ Status:** This is still in *alpha*, and may change without warning.  
<!-- exclude_docs_end -->
<!-- include_docs
:::{important}
**Status:** This project is still in *alpha*, and the API may change without warning.  
:::
include_docs_end -->

<div align="center">

<!-- exclude_docs -->
<!-- <img src="assets/dalle_llama_bias.png" alt="Dalle generated LLama image reading books" style="width: 300px; height: 300px;"> -->
<!-- exclude_docs_end -->
<!-- include_docs
<img src="assets/dalle_llama_bias.png.png" alt="Dalle generated LLama image reading books" style="width: 49%; margin-right: 2%;">
include_docs_end -->

</div>

# Drug Name Perception Analysis: Brand vs Generic

This project looks into how a language model's responses vary between brand names and generic names of drugs. It's an interesting peek into how naming might influence AI-generated perceptions and sentiment.

## What's This All About?

We're curious about whether calling a drug by its brand name vs. its generic name triggers any differences in an AI's responses. Do these differences, if any, suggest a bias towards one naming convention over the other? And what does that say about the knowledge embedded in these AI models?

## Where's the Data From?

We've tapped into the HemOnc database, which is pretty solid when it comes to details on approved cancer drugs. It's our go-to for fetching both brand and generic names of drugs.

## How We're Doing It

We're using GPT-4 (thanks, OpenAI!) to throw questions at it, seeing if how we refer to a drug changes the AI's tune on safety, effectiveness, or side effects. Each drug gets asked the same thing twice, once with its brand name and once with its generic name, to catch any variance.

## Running the Show

Grab the code from GitHub, and make sure you've got the requirements covered with a quick `pip install`.

## Dotenv for API Keys

Keeping API keys out of your code is rule number one. `dotenv` is perfect for this:

1. **Install**: Make sure `python-dotenv` is in your project. If not, `pip install python-dotenv` should do the trick.
2. **Setup**: Drop a `.env` file in your project root with your keys like so:

   ```plaintext
   OPENAI_API_KEY=your_key_here
   ```
3. **Use**: In your code, import `dotenv` and load your keys

   ```python
   from dotenv import load_dotenv
   load_dotenv()
   import os
   ```

   Then, you can access your keys with `os.getenv('OPENAI_API_KEY')`.

4. **Git Ignore**: Don't forget to add `.env` to your `.gitignore` file!


