from fastapi import FastAPI, HTTPException, Query
import os, requests
from dotenv import load_dotenv
import google.generativeai as genai
import re

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")

if not GEMINI_API_KEY or not WOLFRAM_APP_ID:
    raise ValueError("Set GEMINI_API_KEY and WOLFRAM_APP_ID in .env")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="Math Solver + Explanation + Similar Questions")

# -------------------------
# Simple regex-based parsing (more reliable than Gemini for this)
# -------------------------
def parse_question(question: str) -> str:
    """Convert ambiguous math notation to proper format"""
    question = question.strip()
    
    # Pattern 1: "2x 5=15" → "2*x + 5 = 15"
    # Matches: number+variable space number
    question = re.sub(r'(\d+)([a-zA-Z])\s+(\d+)', r'\1*\2 + \3', question)
    
    # Pattern 2: Add * between number and variable if missing
    # "2x" → "2*x"
    question = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', question)
    
    # If it's an equation without "solve", add it
    if "=" in question and not any(word in question.lower() for word in ["solve", "what", "find"]):
        # Try to detect the variable (usually x, y, z)
        variables = re.findall(r'[a-zA-Z]', question)
        var = variables[0] if variables else 'x'
        question = f"solve {question} for {var}"
    
    return question

# -------------------------
# Solve math using WolframAlpha
# -------------------------
def call_wolfram(expr: str):
    url = "http://api.wolframalpha.com/v2/query"
    params = {"input": expr, "appid": WOLFRAM_APP_ID, "output": "JSON", "format": "plaintext"}
    
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        pods = data.get("queryresult", {}).get("pods", [])
        
        if not pods:
            return None
        
        # Priority order for pod titles
        priority_keywords = [
            "solution", "result", "real solution", "solutions", 
            "roots", "value", "decimal approximation", "exact result"
        ]
        
        # First pass: look for priority pods
        for keyword in priority_keywords:
            for pod in pods:
                title = pod.get("title", "").lower()
                if keyword in title:
                    for sub in pod.get("subpods", []):
                        plaintext = sub.get("plaintext", "").strip()
                        if plaintext and plaintext.lower() not in ["false", "true", ""]:
                            return plaintext
        
        # Second pass: get any non-empty result
        for pod in pods:
            title = pod.get("title", "").lower()
            # Skip input interpretation pods
            if "input" not in title:
                for sub in pod.get("subpods", []):
                    plaintext = sub.get("plaintext", "").strip()
                    if plaintext and plaintext.lower() not in ["false", "true", ""]:
                        return plaintext
        
        return None
        
    except Exception as e:
        print(f"WolframAlpha error: {e}")
        return None

# -------------------------
# Generate Gemini explanation
# -------------------------
def gemini_explain(question: str, answer: str):
    prompt = f"""Explain step by step how to solve this math problem.

Question: {question}
Answer: {answer}

Provide a clear, numbered step-by-step solution."""
    
    try:
        resp = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=800,
                candidate_count=1
            )
        )
        
        if resp and resp.text:
            return resp.text.strip()
        else:
            return "Explanation could not be generated."
        
    except Exception as e:
        print(f"Gemini explanation error: {e}")
        return f"Error generating explanation: {str(e)}"

# -------------------------
# Generate similar questions
# -------------------------
def generate_similar_questions(question: str, count: int = 3):
    prompt = f"""Generate {count} math questions similar to: {question}

Just list {count} questions, one per line. Use proper math notation."""
    
    try:
        resp = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=400,
                candidate_count=1
            )
        )
        
        if not resp or not resp.text:
            return ["Could not generate similar questions"]
        
        # Parse response
        lines = resp.text.strip().split("\n")
        similar = []
        
        for line in lines:
            clean = line.strip()
            # Remove numbering patterns
            clean = re.sub(r'^[\d]+[\.\)\:]\s*', '', clean)
            clean = re.sub(r'^[-*•]\s*', '', clean)
            
            if clean and len(clean) > 3 and not clean.startswith("Question"):
                similar.append(clean)
        
        return similar[:count] if similar else ["No similar questions generated"]
        
    except Exception as e:
        print(f"Gemini similar questions error: {e}")
        return [f"Error: {str(e)}"]

# -------------------------
# API endpoint
# -------------------------
@app.get("/solve/")
def solve_math(question: str = Query(..., description="Math question")):
    try:
        # Parse the question
        parsed_question = parse_question(question)
        
        print(f"Original: {question}")
        print(f"Parsed: {parsed_question}")
        
        # Solve with WolframAlpha
        answer = call_wolfram(parsed_question)
        
        if not answer:
            # Try with original question if parsed version fails
            answer = call_wolfram(question)
            
            if not answer:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Could not solve. Try using clear notation like '2*x + 5 = 15'"
                )
        
        # Generate explanation and similar questions
        explanation = gemini_explain(question, answer)
        similar_questions = generate_similar_questions(question)
        
        return {
            "original_question": question,
            "parsed_question": parsed_question,
            "answer": answer,
            "explanation": explanation,
            "similar_questions": similar_questions
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

#
@app.get("/")
def root():
    return {
        "message": "Math Solver API",
        "endpoint": "/solve/",
        "examples": [
            "/solve/?question=2x + 5 = 15",
            "/solve/?question=3*y - 7 = 20",
            "/solve/?question=what is 25 * 47",
            "/solve/?question=integral of x^2"
        ],
        "tips": [
            "Use * for multiplication: 2*x instead of 2x",
            "Use + - * / ^ for operations",
            "For equations: include the = sign"
        ]
    }