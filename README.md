# Code-Reviewer

 A Python code reviewer that checks for syntax errors, potential bugs, and code style issues using common Python linting and analysis tools like flake8, pylint, and a custom syntax checker. The reviewer will take a Python code snippet as input, analyze it, and provide feedback on syntax errors

# Obtain API Keys

Gemini:

Go to Google AI Studio.
Sign in with a Google account.
Create a new API key or copy an existing one.


OpenAI:

Go to OpenAI Platform.
Sign in, navigate to API keys, and generate a new key.


Note: You only need one key (Gemini or OpenAI) to start the service, but you can set both if you plan to use both LLMs.

# Steps to Run the Code
Install required packages: 
pip install fastapi uvicorn black google-generativeai openai requests python-multipart.
