#By Sreehari
import ast
import os
import json
import requests
import black
import google.generativeai as genai
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import io

app = FastAPI(title="Python Code Reviewer Service", description="Analyze Python code with LLM for bugs, efficiency, and idiomatic improvements")

# Serve static files (HTML)
app.mount("/static", StaticFiles(directory="."), name="static")

def configure_gemini(api_key: str):
    """Configure Gemini API with the provided key."""
    try:
        genai.configure(api_key=api_key)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": f"Gemini API configuration error: {str(e)}"}

def configure_openai(api_key: str):
    """Configure OpenAI API with the provided key."""
    try:
        client = OpenAI(api_key=api_key)
        return {"status": "success", "client": client}
    except Exception as e:
        return {"status": "error", "message": f"OpenAI API configuration error: {str(e)}"}

def check_line_limit(code: str, max_lines: int = 500):
    """Check if code exceeds the maximum line limit."""
    line_count = len(code.splitlines())
    if line_count > max_lines:
        return {"status": "error", "message": f"Code exceeds {max_lines} lines ({line_count} lines detected)"}
    return {"status": "success", "line_count": line_count}

def format_code(code: str):
    """Format Python code using black."""
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        return {"status": "success", "code": formatted_code}
    except Exception as e:
        return {"status": "error", "message": f"Black formatting error: {str(e)}"}

def check_syntax(code: str):
    """Check for syntax errors in the provided Python code."""
    try:
        ast.parse(code)
        return {"status": "success", "errors": []}
    except SyntaxError as e:
        return {"status": "error", "errors": [f"SyntaxError: {str(e)} at line {e.lineno}"]}

def analyze_with_llm(code: str, llm_provider: str, model: str, api_key: str):
    prompt = f"""
    You are an expert Python code reviewer. Analyze the following Python code (max 500 lines) and provide a detailed, structured response focusing on:
    1. **Bugs and Issues**: Identify potential bugs, logical errors, edge cases, or runtime issues.
    2. **Cleaner/More Efficient Alternatives**: Suggest optimizations for performance, memory usage, or cleaner code structure.
    3. **Non-Idiomatic or Hard-to-Read Code**: Point out code that is not Pythonic, overly complex, or hard to read, with suggestions for improvement.
    For each issue, provide:
    - The specific line number(s) affected.
    - A clear explanation of the problem.
    - Actionable advice to fix or improve it.
    Return the response in a structured JSON format with three keys: `bugs`, `efficiency`, and `idiomatic`, each containing a list of objects with `line`, `issue`, and `advice` fields. Ensure the response is valid JSON.
    Code:
    ```python
    {code}
    ```
    """
    
    if llm_provider == "gemini":
        try:
            client = genai.GenerativeModel(model)
            response = client.generate_content(prompt)
            return {"status": "success", "suggestions": response.text}
        except Exception as e:
            return {"status": "error", "message": f"Gemini analysis error: {str(e)}"}
    
    elif llm_provider == "openai":
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return {"status": "success", "suggestions": response.choices[0].message.content}
        except Exception as e:
            return {"status": "error", "message": f"OpenAI analysis error: {str(e)}"}
    
    else:
        return {"status": "error", "message": f"Unsupported LLM provider: {llm_provider}"}

def fetch_github_pr(repo: str, pr_number: str, token: str):
    """Fetch code changes from a GitHub pull request."""
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.diff"}
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        diff = response.text
        code = ""
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++") and ".py" in diff:
                code += line[1:] + "\n"
        return {"status": "success", "code": code}
    except Exception as e:
        return {"status": "error", "message": f"Error fetching PR: {str(e)}"}

def review_code(code: str, llm_provider: str, model: str, api_key: str):
    result = {"line_limit": [], "formatting": [], "syntax": [], "bugs": [], "efficiency": [], "idiomatic": []}
    line_check = check_line_limit(code)
    if line_check["status"] == "error":
        result["line_limit"] = [line_check["message"]]
        return result
    format_result = format_code(code)
    if format_result["status"] == "error":
        result["formatting"] = [format_result["message"]]
        return result
    formatted_code = format_result["code"]
    syntax_result = check_syntax(formatted_code)
    if syntax_result["status"] == "error":
        result["syntax"] = syntax_result["errors"]
        return result
    llm_result = analyze_with_llm(formatted_code, llm_provider, model, api_key)
    if llm_result["status"] == "error":
        result["bugs"] = [llm_result["message"]]
    else:
        try:
            suggestions = json.loads(llm_result["suggestions"])
            print("Parsed Suggestions:", suggestions)
            result["bugs"] = suggestions.get("bugs", [])
            result["efficiency"] = suggestions.get("efficiency", [])
            result["idiomatic"] = suggestions.get("idiomatic", [])
        except json.JSONDecodeError as e:
            print("JSON Parse Error:", str(e), "Raw Response:", llm_result["suggestions"])
            result["bugs"] = [f"Error parsing LLM response: {llm_result['suggestions']}"]
    return result

class GitHubPRRequest(BaseModel):
    repo: str
    pr_number: str
    token: str
    llm_provider: str = "gemini"
    model: Optional[str] = "gemini-2.5-flash"
    api_key: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
    print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    """Configure LLM APIs on startup."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not (gemini_api_key or openai_api_key):
        raise HTTPException(status_code=500, detail="At least one LLM API key (GEMINI_API_KEY or OPENAI_API_KEY) must be configured")
    if gemini_api_key:
        gemini_config = configure_gemini(gemini_api_key)
        if gemini_config["status"] == "error":
            raise HTTPException(status_code=500, detail=gemini_config["message"])

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI."""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...), llm_provider: str = "gemini", model: Optional[str] = "gemini-2.5-flash", api_key: Optional[str] = None):
    """Analyze uploaded Python file."""
    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="File must be a .py file")
    
    # Determine API key
    if llm_provider == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY")
    elif llm_provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {llm_provider}")
    
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key for {llm_provider} is required")
    
    try:
        code = await file.read()
        code = code.decode("utf-8")
        report = review_code(code, llm_provider, model, api_key)
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/analyze/github")
async def analyze_github_pr(request: GitHubPRRequest):
    """Analyze code from a GitHub PR."""
    # Determine API key
    if request.llm_provider == "gemini":
        api_key = request.api_key or os.getenv("GEMINI_API_KEY")
    elif request.llm_provider == "openai":
        api_key = request.api_key or os.getenv("OPENAI_API_KEY")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {request.llm_provider}")
    
    if not api_key:
        raise HTTPException(status_code=400, detail=f"API key for {request.llm_provider} is required")
    
    pr_result = fetch_github_pr(request.repo, request.pr_number, request.token)
    if pr_result["status"] == "error":
        raise HTTPException(status_code=400, detail=pr_result["message"])
    report = review_code(pr_result["code"], request.llm_provider, request.model, api_key)
    return JSONResponse(content=report)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
