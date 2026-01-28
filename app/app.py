from fastapi import FastAPI, HTTPException, Query
import os, requests, re
from dotenv import load_dotenv
import google.generativeai as genai
from sympy import symbols, Eq, solve, SympifyError, sympify

# -------------------------
# Load environment
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")

if not GEMINI_API_KEY or not WOLFRAM_APP_ID:
    raise ValueError("Set GEMINI_API_KEY and WOLFRAM_APP_ID in .env")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="Enhanced Math Solver + Explanation + Similar Questions")

# -------------------------
# Parse question for simple fixes
# -------------------------
def parse_question(question: str) -> str:
    question = question.strip()
    # "2x 5=15" → "2*x + 5 = 15"
    question = re.sub(r'(\d+)([a-zA-Z])\s+(\d+)', r'\1*\2 + \3', question)
    # "2x" → "2*x"
    question = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', question)
    # Add "solve" if equation
    if "=" in question and not any(word in question.lower() for word in ["solve", "what", "find"]):
        variables = re.findall(r'[a-zA-Z]', question)
        var = variables[0] if variables else 'x'
        question = f"solve {question} for {var}"
    return question

# -------------------------
# WolframAlpha API
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
        priority_keywords = [
            "solution", "result", "real solution", "solutions", 
            "roots", "value", "decimal approximation", "exact result"
        ]
        # Priority pass
        for keyword in priority_keywords:
            for pod in pods:
                title = pod.get("title", "").lower()
                if keyword in title:
                    for sub in pod.get("subpods", []):
                        plaintext = sub.get("plaintext", "").strip()
                        if plaintext and plaintext.lower() not in ["false", "true", ""]:
                            return plaintext
        # Any non-empty pod
        for pod in pods:
            title = pod.get("title", "").lower()
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
# Gemini explanation
# -------------------------
def gemini_explain(question: str, answer: str, candidate_count: int = 1):
    prompt = f"""
Explain step by step how to solve this math problem.
Include all intermediate steps and algebraic simplifications.

Question: {question}
Answer: {answer}

Provide a clear, numbered step-by-step solution.
"""
    try:
        resp = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2000,
                candidate_count=candidate_count
            )
        )
        if resp and resp.text:
            return resp.text.strip()
        return "Explanation could not be generated."
    except Exception as e:
        print(f"Gemini explanation error: {e}")
        return f"Error generating explanation: {str(e)}"

# -------------------------
# Gemini similar questions
# -------------------------
def generate_similar_questions(question: str, count: int = 3):
    prompt = f"""
Generate {count} math questions similar to: {question}
Just list {count} questions, one per line. Use proper math notation.
"""
    try:
        resp = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=1000,
                candidate_count=1
            )
        )
        if not resp or not resp.text:
            return ["Could not generate similar questions"]
        lines = resp.text.strip().split("\n")
        similar = []
        for line in lines:
            clean = re.sub(r'^[\d]+[\.\)\:]\s*', '', line.strip())
            clean = re.sub(r'^[-*•]\s*', '', clean)
            if clean and len(clean) > 3 and not clean.startswith("Question"):
                similar.append(clean)
        return similar[:count] if similar else ["No similar questions generated"]
    except Exception as e:
        print(f"Gemini similar questions error: {e}")
        return [f"Error: {str(e)}"]

# -------------------------
# Detect difficulty & topic
# -------------------------
def detect_topic_difficulty(question: str):
    prompt = f"""
Classify the following math question into topic (Algebra, Calculus, Trigonometry, etc.)
and difficulty (Easy, Medium, Hard):

Question: {question}
"""
    try:
        resp = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=200,
                candidate_count=1
            )
        )
        if resp and resp.text:
            return resp.text.strip()
        return "Could not detect topic/difficulty"
    except Exception as e:
        print(f"Gemini topic/difficulty error: {e}")
        return f"Error: {str(e)}"

# -------------------------
# SymPy local solver fallback
# -------------------------
def solve_locally(expr: str):
    try:
        # Extract variable
        variables = re.findall(r'[a-zA-Z]', expr)
        var = symbols(variables[0]) if variables else symbols('x')
        equation = sympify(expr.replace('=', '-(') + ')')
        solution = solve(equation, var)
        return solution
    except (SympifyError, Exception) as e:
        print(f"SymPy error: {e}")
        return None

# -------------------------
# API endpoints
# -------------------------
@app.get("/solve/")
def solve_math(question: str = Query(..., description="Math question")):
    try:
        parsed_question = parse_question(question)
        answer = call_wolfram(parsed_question)
        if not answer:
            answer = call_wolfram(question)
        if not answer:
            # Fallback local solver
            local_ans = solve_locally(parsed_question)
            if local_ans is not None:
                answer = str(local_ans)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not solve. Try using clear notation like '2*x + 5 = 15'"
                )
        explanation = gemini_explain(question, answer, candidate_count=1)
        similar_questions = generate_similar_questions(question)
        topic_difficulty = detect_topic_difficulty(question)
        return {
            "original_question": question,
            "parsed_question": parsed_question,
            "answer": answer,
            "explanation": explanation,
            "similar_questions": similar_questions,
            "topic_difficulty": topic_difficulty
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/solve-simple/")
def solve_simple(expression: str = Query(..., description="Direct math expression like '2+2'")):
    try:
        answer = call_wolfram(expression)
        if not answer:
            # Local SymPy fallback
            local_ans = solve_locally(expression)
            if local_ans is not None:
                answer = str(local_ans)
            else:
                raise HTTPException(status_code=400, detail="Could not solve the expression")
        return {"expression": expression, "answer": answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Enhanced Math Solver API",
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
