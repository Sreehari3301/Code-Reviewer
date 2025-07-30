# Code-Reviewer

 A Python code reviewer that checks for syntax errors, potential bugs, and code style issues using common Python linting and analysis tools like flake8, pylint, and a custom syntax checker. The reviewer will take a Python code snippet as input, analyze it, and provide feedback on syntax errors

Gemini:

Go to Google AI Studio.
Sign in with a Google account.
Create a new API key or copy an existing one.


OpenAI:

Go to OpenAI Platform.
Sign in, navigate to API keys, and generate a new key.


Note: You only need one key (Gemini or OpenAI) to start the service, but you can set both if you plan to use both LLMs.

# Steps to Run the Code


# First Download the Repo in your system

select the folder

1. **Clone the repository**

   ```bash
   git clone https://github.com/gamkers/Project_Jarvis.git
   cd Project_Jarvis
   ```

2. **Install Python dependencies**
   ```bash
   pip install fastapi uvicorn black google-generativeai openai requests python-multipart.
   ```

3. **Set API Keys**

   - Get your Gemini API key here: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - Get your OpenAI API key 

   Export your API keys as environment variables:

   ```bash
   export GEMINI_API_KEY='your_gemini_api_key'
   export OPENAI_API_KEY='your_openai_api_key'
   ```
OR

   ```bash
   set GEMINI_API_KEY='your_gemini_api_key'
   set OPENAI_API_KEY='your_openai_api_key'
   ```

4.**Running the Service**

- Navigate to the directory containing code_reviewer_service.py

```bash
cd code_reviewer
```
- Run the server

```bash
python code_reviewer_service.py
```
5. **Access the Web UI**
- Create code.py (the code you wanted to check) in the downloaded folder.
- Open a browser and navigate to http://localhost:8000.
- The UI displays:
- Click "Choose File" and select code.py.
- Verify the file name appears.
- Select gemini, enter gemini-2.5-flash.
- Leave "LLM API Key" blank if $env:GEMINI_API_KEY is set; otherwise, paste your key.
- Click "Analyze Code."
