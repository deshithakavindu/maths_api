from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import os, requests, re, json, hashlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from sympy import symbols, Eq, solve, SympifyError, sympify, latex, simplify
import asyncio
import aiohttp
from functools import lru_cache
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from enum import Enum

# -------------------------
# Load environment
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")

if not GEMINI_API_KEY or not WOLFRAM_APP_ID:
    raise ValueError("Set GEMINI_API_KEY and WOLFRAM_APP_ID in .env")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# -------------------------
# Simple in-memory cache
# -------------------------
class SimpleCache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        self.cache.clear()

cache = SimpleCache(ttl_seconds=3600)  # 1 hour cache

# -------------------------
# Pydantic Models
# -------------------------
class DifficultyLevel(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    EXPERT = "Expert"

class MathTopic(str, Enum):
    ALGEBRA = "Algebra"
    CALCULUS = "Calculus"
    TRIGONOMETRY = "Trigonometry"
    GEOMETRY = "Geometry"
    STATISTICS = "Statistics"
    LINEAR_ALGEBRA = "Linear Algebra"
    OTHER = "Other"

class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    include_graph: bool = Field(default=False, description="Generate visualization if applicable")
    include_latex: bool = Field(default=True, description="Include LaTeX formatted answer")
    hint_mode: bool = Field(default=False, description="Return hints instead of full solution")
    similar_count: int = Field(default=3, ge=1, le=10, description="Number of similar questions")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        # Check for potentially dangerous characters
        dangerous_chars = ['<', '>', '{', '}', ';', '`']
        if any(char in v for char in dangerous_chars):
            raise ValueError("Question contains invalid characters")
        return v.strip()

class BatchSolveRequest(BaseModel):
    questions: List[str] = Field(..., min_items=1, max_items=10)
    include_explanations: bool = Field(default=True)

class SolveResponse(BaseModel):
    original_question: str
    parsed_question: str
    answer: str
    latex_answer: Optional[str] = None
    explanation: Optional[str] = None
    step_by_step: Optional[List[str]] = None
    similar_questions: List[str]
    topic: str
    difficulty: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    graph_base64: Optional[str] = None
    alternative_methods: Optional[List[str]] = None
    computation_time_ms: float

# -------------------------
# FastAPI App Setup
# -------------------------
app = FastAPI(
    title="Enhanced Math Solver API v2",
    description="Advanced math solver with AI explanations, visualizations, and learning features",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helper Functions
# -------------------------
def generate_cache_key(prefix: str, data: str) -> str:
    """Generate cache key from data"""
    return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"

def parse_question(question: str) -> str:
    """Enhanced question parsing with better notation handling"""
    question = question.strip()
    
    # Handle implicit multiplication: "2x 5=15" → "2*x + 5 = 15"
    question = re.sub(r'(\d+)([a-zA-Z])\s+(\d+)', r'\1*\2 + \3', question)
    
    # Handle "2x" → "2*x"
    question = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', question)
    
    # Handle "x(x+1)" → "x*(x+1)"
    question = re.sub(r'([a-zA-Z])\(', r'\1*(', question)
    question = re.sub(r'\)(\d)', r')*\1', question)
    question = re.sub(r'\)([a-zA-Z])', r')*\1', question)
    
    # Handle exponents: "x^2" is fine, but "x**2" also works
    question = question.replace('^', '**')
    
    # Add "solve" prefix if equation without instruction
    if "=" in question and not any(word in question.lower() for word in ["solve", "what", "find", "calculate"]):
        variables = re.findall(r'[a-zA-Z]', question)
        var = variables[0] if variables else 'x'
        question = f"solve {question} for {var}"
    
    return question

async def call_wolfram_async(expr: str) -> Optional[str]:
    """Async WolframAlpha API call with better error handling"""
    cache_key = generate_cache_key("wolfram", expr)
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    url = "http://api.wolframalpha.com/v2/query"
    params = {
        "input": expr,
        "appid": WOLFRAM_APP_ID,
        "output": "JSON",
        "format": "plaintext"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                pods = data.get("queryresult", {}).get("pods", [])
                
                if not pods:
                    return None
                
                # Priority keywords for finding the answer
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
                                    cache.set(cache_key, plaintext)
                                    return plaintext
                
                # Second pass: any non-input pod
                for pod in pods:
                    title = pod.get("title", "").lower()
                    if "input" not in title:
                        for sub in pod.get("subpods", []):
                            plaintext = sub.get("plaintext", "").strip()
                            if plaintext and plaintext.lower() not in ["false", "true", ""]:
                                cache.set(cache_key, plaintext)
                                return plaintext
                
                return None
                
    except asyncio.TimeoutError:
        print(f"WolframAlpha timeout for: {expr}")
        return None
    except Exception as e:
        print(f"WolframAlpha error: {e}")
        return None

def solve_locally(expr: str) -> Optional[Dict[str, Any]]:
    """Enhanced local solver with SymPy"""
    try:
        # Extract variables
        variables = list(set(re.findall(r'[a-zA-Z]', expr)))
        if not variables:
            # Try to evaluate as expression
            result = sympify(expr)
            return {
                "solution": str(result),
                "latex": latex(result),
                "method": "evaluation"
            }
        
        var = symbols(variables[0])
        
        if '=' in expr:
            left, right = expr.split('=', 1)
            equation = sympify(left) - sympify(right)
            solutions = solve(equation, var)
        else:
            solutions = solve(sympify(expr), var)
        
        if not solutions:
            return None
        
        # Format solution
        if len(solutions) == 1:
            sol = solutions[0]
            return {
                "solution": str(sol),
                "latex": latex(sol),
                "method": "symbolic"
            }
        else:
            return {
                "solution": str(solutions),
                "latex": latex(solutions),
                "method": "symbolic",
                "multiple": True
            }
            
    except (SympifyError, Exception) as e:
        print(f"SymPy error: {e}")
        return None

async def gemini_explain_async(question: str, answer: str, hint_mode: bool = False) -> str:
    """Async Gemini explanation with hint mode support"""
    cache_key = generate_cache_key(f"explain_{hint_mode}", f"{question}:{answer}")
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    if hint_mode:
        prompt = f"""
Provide helpful hints for solving this math problem WITHOUT giving away the complete answer.
Give 2-3 progressive hints that guide the student toward the solution.

Question: {question}
Expected Answer: {answer}

Format your response as numbered hints, each building on the previous one.
"""
    else:
        prompt = f"""
Explain step by step how to solve this math problem.
Include all intermediate steps and algebraic simplifications.
Use clear mathematical notation.

Question: {question}
Answer: {answer}

Provide a clear, numbered step-by-step solution.
End with verification by substituting the answer back into the original equation if applicable.
"""
    
    try:
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3 if not hint_mode else 0.5,
                max_output_tokens=2000,
                candidate_count=1
            )
        )
        
        if response and response.text:
            result = response.text.strip()
            cache.set(cache_key, result)
            return result
        return "Explanation could not be generated."
        
    except Exception as e:
        print(f"Gemini explanation error: {e}")
        return f"Error generating explanation: {str(e)}"

async def generate_similar_questions_async(question: str, count: int = 3) -> List[str]:
    """Async similar questions generation"""
    cache_key = generate_cache_key("similar", f"{question}:{count}")
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    prompt = f"""
Generate {count} math questions similar in difficulty and topic to: {question}
Make them educational and slightly varied in approach.
Just list {count} questions, one per line. Use proper math notation.
"""
    
    try:
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=1000,
                candidate_count=1
            )
        )
        
        if not response or not response.text:
            return ["Could not generate similar questions"]
        
        lines = response.text.strip().split("\n")
        similar = []
        
        for line in lines:
            # Clean up numbering and formatting
            clean = re.sub(r'^[\d]+[\.\)\:]\s*', '', line.strip())
            clean = re.sub(r'^[-*•]\s*', '', clean)
            if clean and len(clean) > 3 and not clean.lower().startswith("question"):
                similar.append(clean)
        
        result = similar[:count] if similar else ["No similar questions generated"]
        cache.set(cache_key, result)
        return result
        
    except Exception as e:
        print(f"Gemini similar questions error: {e}")
        return [f"Error: {str(e)}"]

async def detect_topic_difficulty_async(question: str) -> Dict[str, str]:
    """Async topic and difficulty detection"""
    cache_key = generate_cache_key("topic_diff", question)
    cached = cache.get(cache_key)
    if cached:
        return cached
    
    prompt = f"""
Classify the following math question:
1. Topic: Choose from (Algebra, Calculus, Trigonometry, Geometry, Statistics, Linear Algebra, Other)
2. Difficulty: Choose from (Easy, Medium, Hard, Expert)

Question: {question}

Respond in this exact format:
Topic: <topic>
Difficulty: <difficulty>
"""
    
    try:
        response = await asyncio.to_thread(
            gemini_model.generate_content,
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=100,
                candidate_count=1
            )
        )
        
        if response and response.text:
            text = response.text.strip()
            topic = "Other"
            difficulty = "Medium"
            
            # Parse response
            for line in text.split("\n"):
                if "topic:" in line.lower():
                    topic = line.split(":", 1)[1].strip()
                elif "difficulty:" in line.lower():
                    difficulty = line.split(":", 1)[1].strip()
            
            result = {"topic": topic, "difficulty": difficulty}
            cache.set(cache_key, result)
            return result
        
        return {"topic": "Other", "difficulty": "Medium"}
        
    except Exception as e:
        print(f"Gemini topic/difficulty error: {e}")
        return {"topic": "Other", "difficulty": "Medium"}

def generate_graph(expression: str, variable: str = 'x', x_range: tuple = (-10, 10)) -> Optional[str]:
    """Generate graph visualization and return base64 encoded image"""
    try:
        var = symbols(variable)
        expr = sympify(expression)
        
        # Generate x values
        x_vals = np.linspace(x_range[0], x_range[1], 500)
        
        # Evaluate expression
        func = lambda x_val: float(expr.subs(var, x_val))
        y_vals = [func(x_val) for x_val in x_vals]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.axvline(x=0, color='k', linewidth=0.5)
        plt.xlabel(variable, fontsize=12)
        plt.ylabel(f'f({variable})', fontsize=12)
        plt.title(f'Graph of {expression}', fontsize=14)
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Graph generation error: {e}")
        return None

def calculate_confidence(source: str, answer: str) -> float:
    """Calculate confidence score based on source and answer quality"""
    confidence = 0.5  # Base confidence
    
    if source == "wolfram":
        confidence = 0.95
    elif source == "sympy":
        confidence = 0.85
    elif source == "gemini":
        confidence = 0.75
    
    # Adjust based on answer characteristics
    if answer and answer.lower() not in ["none", "error", "could not solve"]:
        confidence += 0.05
    
    # Check if answer is numeric
    if re.search(r'\d', answer):
        confidence += 0.05
    
    return min(confidence, 1.0)

# -------------------------
# API Endpoints
# -------------------------
@app.get("/")
def root():
    """API documentation and examples"""
    return {
        "message": "Enhanced Math Solver API v2.0",
        "features": [
            "AI-powered explanations",
            "Step-by-step solutions",
            "Graph visualizations",
            "LaTeX formatting",
            "Similar question generation",
            "Hint mode for learning",
            "Batch processing",
            "Caching for performance"
        ],
        "endpoints": {
            "/solve/": "Main solver endpoint with all features",
            "/solve-simple/": "Quick calculation endpoint",
            "/solve-batch/": "Batch solve multiple questions",
            "/hint/": "Get hints without full solution",
            "/verify/": "Verify an answer to a question",
            "/health": "API health check"
        },
        "examples": [
            "/solve/?question=2x + 5 = 15",
            "/solve/?question=integrate x^2 dx&include_graph=true",
            "/solve/?question=solve x^2 - 5x + 6 = 0&include_latex=true",
            "/hint/?question=3y - 7 = 20",
            "/solve-simple/?expression=sqrt(144)"
        ],
        "tips": [
            "Use * for multiplication: 2*x instead of 2x",
            "Use ** or ^ for exponents: x^2 or x**2",
            "Enable graph generation with include_graph=true",
            "Get hints with hint_mode=true or use /hint/ endpoint"
        ]
    }

@app.get("/solve/", response_model=SolveResponse)
async def solve_math(
    question: str = Query(..., description="Math question to solve"),
    include_graph: bool = Query(False, description="Generate visualization"),
    include_latex: bool = Query(True, description="Include LaTeX formatting"),
    hint_mode: bool = Query(False, description="Return hints instead of solution"),
    similar_count: int = Query(3, ge=1, le=10, description="Number of similar questions")
):
    """
    Main solver endpoint with comprehensive features.
    Supports algebra, calculus, trigonometry, and more.
    """
    start_time = datetime.now()
    
    try:
        # Validate and parse question
        parsed_question = parse_question(question)
        
        # Parallel async operations
        wolfram_task = call_wolfram_async(parsed_question)
        topic_diff_task = detect_topic_difficulty_async(question)
        similar_task = generate_similar_questions_async(question, similar_count)
        
        # Gather results
        answer, topic_diff, similar_questions = await asyncio.gather(
            wolfram_task,
            topic_diff_task,
            similar_task
        )
        
        source = "wolfram"
        latex_answer = None
        
        # Fallback to local solver if Wolfram fails
        if not answer:
            answer_fallback = call_wolfram_async(question)
            if answer_fallback:
                answer = await answer_fallback
            else:
                local_result = solve_locally(parsed_question)
                if local_result:
                    answer = local_result["solution"]
                    latex_answer = local_result["latex"]
                    source = "sympy"
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not solve. Try rephrasing or using clearer notation like '2*x + 5 = 15'"
                    )
        
        # Generate explanation
        explanation = await gemini_explain_async(question, answer, hint_mode)
        
        # Generate LaTeX if requested and not already available
        if include_latex and not latex_answer:
            try:
                latex_answer = latex(sympify(answer))
            except:
                latex_answer = answer
        
        # Generate graph if requested
        graph_base64 = None
        if include_graph and not hint_mode:
            try:
                # Extract expression for graphing
                if '=' in parsed_question:
                    expr = parsed_question.split('=')[0].replace('solve', '').strip()
                else:
                    expr = parsed_question
                
                graph_base64 = generate_graph(expr)
            except Exception as e:
                print(f"Graph generation skipped: {e}")
        
        # Calculate computation time
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate confidence score
        confidence = calculate_confidence(source, answer)
        
        return SolveResponse(
            original_question=question,
            parsed_question=parsed_question,
            answer=answer,
            latex_answer=latex_answer,
            explanation=explanation if not hint_mode else None,
            step_by_step=[explanation] if hint_mode else None,
            similar_questions=similar_questions,
            topic=topic_diff["topic"],
            difficulty=topic_diff["difficulty"],
            confidence_score=confidence,
            graph_base64=graph_base64,
            computation_time_ms=round(computation_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in solve_math: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/solve-simple/")
async def solve_simple(
    expression: str = Query(..., description="Direct math expression like '2+2' or 'sqrt(16)'")
):
    """
    Quick calculation endpoint for simple expressions.
    No AI explanations, just fast computation.
    """
    try:
        # Check cache first
        cache_key = generate_cache_key("simple", expression)
        cached = cache.get(cache_key)
        if cached:
            return {"expression": expression, "answer": cached, "cached": True}
        
        # Try Wolfram first
        answer = await call_wolfram_async(expression)
        
        # Fallback to local evaluation
        if not answer:
            local_result = solve_locally(expression)
            if local_result:
                answer = local_result["solution"]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Could not evaluate the expression"
                )
        
        cache.set(cache_key, answer)
        
        return {
            "expression": expression,
            "answer": answer,
            "cached": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/solve-batch/")
async def solve_batch(request: BatchSolveRequest):
    """
    Solve multiple questions in one request.
    Useful for homework sets or practice problems.
    """
    try:
        results = []
        
        for question in request.questions:
            try:
                parsed = parse_question(question)
                answer = await call_wolfram_async(parsed)
                
                if not answer:
                    local_result = solve_locally(parsed)
                    answer = local_result["solution"] if local_result else "Could not solve"
                
                result = {
                    "question": question,
                    "answer": answer,
                    "success": True
                }
                
                if request.include_explanations:
                    explanation = await gemini_explain_async(question, answer)
                    result["explanation"] = explanation
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "question": question,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "total_questions": len(request.questions),
            "successful": sum(1 for r in results if r.get("success", False)),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/hint/")
async def get_hint(question: str = Query(..., description="Math question")):
    """
    Get hints for solving a problem without revealing the full answer.
    Great for learning and practice.
    """
    try:
        parsed_question = parse_question(question)
        
        # Get the answer first (but don't return it)
        answer = await call_wolfram_async(parsed_question)
        if not answer:
            local_result = solve_locally(parsed_question)
            answer = local_result["solution"] if local_result else "Unknown"
        
        # Generate hints
        hints = await gemini_explain_async(question, answer, hint_mode=True)
        
        return {
            "question": question,
            "hints": hints,
            "message": "Try to solve it yourself first, then check your answer!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/verify/")
async def verify_answer(
    question: str = Query(..., description="The math question"),
    user_answer: str = Query(..., description="Your proposed answer")
):
    """
    Verify if a user's answer is correct.
    Useful for self-checking homework.
    """
    try:
        parsed_question = parse_question(question)
        
        # Get correct answer
        correct_answer = await call_wolfram_async(parsed_question)
        if not correct_answer:
            local_result = solve_locally(parsed_question)
            correct_answer = local_result["solution"] if local_result else None
        
        if not correct_answer:
            return {
                "question": question,
                "user_answer": user_answer,
                "verified": False,
                "message": "Could not verify - unable to solve question"
            }
        
        # Normalize answers for comparison
        def normalize(ans):
            return re.sub(r'\s+', '', str(ans).lower())
        
        is_correct = normalize(user_answer) == normalize(correct_answer)
        
        # Try numerical comparison if string comparison fails
        if not is_correct:
            try:
                user_val = float(sympify(user_answer))
                correct_val = float(sympify(correct_answer))
                is_correct = abs(user_val - correct_val) < 0.0001
            except:
                pass
        
        return {
            "question": question,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "message": "Correct! Well done!" if is_correct else "Not quite. Try again!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(cache.cache),
        "version": "2.0.0"
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear the API cache (admin endpoint)"""
    cache.clear()
    return {"message": "Cache cleared successfully", "timestamp": datetime.now().isoformat()}

# -------------------------
# Error handlers
# -------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

