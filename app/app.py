from fastapi import FastAPI, HTTPException, Query
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")

if not GEMINI_API_KEY or not WOLFRAM_APP_ID:
    raise ValueError("Please set GEMINI_API_KEY and WOLFRAM_APP_ID in .env")

# Configure Gemini using official SDK
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI(title="Math Solver with Gemini + WolframAlpha")

# Rate limiting
last_gemini_call = 0
RATE_LIMIT_DELAY = 1  # seconds between calls

# -------------------------
# Helper function: Gemini API call using SDK
# -------------------------
@lru_cache(maxsize=100)
def call_gemini(prompt: str):
    global last_gemini_call
    
    # Rate limiting
    current_time = time.time()
    time_since_last = current_time - last_gemini_call
    if time_since_last < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - time_since_last)
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256
            )
        )
        last_gemini_call = time.time()
        return response.text.strip()
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again in a few seconds."
            )
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {error_msg}")

# -------------------------
# Helper function: WolframAlpha API call
# -------------------------
@lru_cache(maxsize=100)
def call_wolfram(expression: str):
    url = "http://api.wolframalpha.com/v2/query"
    params = {
        "input": expression,
        "appid": WOLFRAM_APP_ID,
        "output": "JSON",
        "format": "plaintext"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("queryresult", {}).get("success"):
            return None
        
        # Extract result from pods
        pods = data["queryresult"].get("pods", [])
        result_text = ""
        
        for pod in pods:
            title = pod.get("title", "").lower()
            if any(keyword in title for keyword in ["result", "solution", "value", "integral", "decimal approximation"]):
                for sub in pod.get("subpods", []):
                    plaintext = sub.get("plaintext")
                    if plaintext:
                        result_text = plaintext
                        break
                if result_text:
                    break
        
        return result_text if result_text else None
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"WolframAlpha Error: {str(e)}")

# -------------------------
# Main solve endpoint
# -------------------------
@app.get("/solve/")
def solve_math(
    question: str = Query(..., description="Math question like 'What is 2+2?'"),
    include_explanation: bool = Query(True, description="Include step-by-step explanation")
):
    try:
        # Extract clean math expression using basic parsing first
        math_expression = question.strip()
        
        # Common patterns
        if "what is" in question.lower():
            math_expression = question.lower().replace("what is", "").strip().rstrip("?")
        elif "solve" in question.lower():
            math_expression = question.lower().replace("solve", "").strip().rstrip("?")
        
        # Use WolframAlpha to solve
        answer = call_wolfram(math_expression)
        
        if not answer:
            # If WolframAlpha fails, try using Gemini to parse the expression
            gemini_prompt = f"""Extract only the mathematical expression from this question. 
Return ONLY the math expression with no extra text.
Question: {question}
Expression:"""
            
            math_expression = call_gemini(gemini_prompt)
            answer = call_wolfram(math_expression)
            
            if not answer:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not solve the expression. Please rephrase your question."
                )
        
        response = {
            "original_question": question,
            "parsed_expression": math_expression,
            "answer": answer
        }
        
        # Only call Gemini for explanation if requested
        if include_explanation:
            explanation_prompt = f"""Briefly explain how to solve: {math_expression}
Answer: {answer}
Keep it concise (2-3 sentences)."""
            
            explanation = call_gemini(explanation_prompt)
            response["explanation"] = explanation
        
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# -------------------------
# Simple solve endpoint (no Gemini, just WolframAlpha)
# -------------------------
@app.get("/solve-simple/")
def solve_simple(expression: str = Query(..., description="Direct math expression like '2+2'")):
    """Solve math directly without AI parsing - faster and no rate limits"""
    try:
        answer = call_wolfram(expression)
        
        if not answer:
            raise HTTPException(status_code=400, detail="Could not solve the expression")
        
        return {
            "expression": expression,
            "answer": answer
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# -------------------------
# Ask Gemini directly (like your other project)
# -------------------------
@app.get("/ask/")
def ask_gemini(question: str = Query(..., description="Ask Gemini anything")):
    """
    Ask Gemini any question directly
    Example: /ask/?question=What is Python?
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer = call_gemini(question)
        return {
            "question": question,
            "answer": answer
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Math Solver API (Gemini SDK + WolframAlpha)",
        "endpoints": {
            "/solve/": "AI-powered math solver with explanations",
            "/solve-simple/": "Direct math solver (faster, no AI parsing)",
            "/ask/": "Ask Gemini any question"
        },
        "examples": {
            "math_with_explanation": "/solve/?question=What is 2+2",
            "math_no_explanation": "/solve/?question=2+2&include_explanation=false",
            "direct_solve": "/solve-simple/?expression=25*47",
            "ask_anything": "/ask/?question=Explain quantum computing"
        }
    }