"""
app.py - Gradio web interface for the GAIA Quiz Agent
"""
import os
import requests
import gradio as gr
from typing import List, Dict, Optional
from dotenv import load_dotenv
from smolagents.agents import ToolCallingAgent
from smolagents.tools import Tool
from litellm import completion
from agent import GAIAAgent

# Load environment variables
load_dotenv()

# API endpoints
BASE_URL = "https://gaia-quiz-api.huggingface.co"
QUESTIONS_ENDPOINT = f"{BASE_URL}/questions"
SUBMIT_ENDPOINT = f"{BASE_URL}/submit"

# Define custom tools as subclasses of Tool
class SearchWebTool(Tool):
    name = "search_web"
    description = "Search the web for information."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                return "\n".join([f"{r['title']}: {r['body']}" for r in results])
        except Exception as e:
            return f"Error searching web: {str(e)}"

class SearchWikipediaTool(Tool):
    name = "search_wikipedia"
    description = "Search Wikipedia for information."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            import wikipedia
            return wikipedia.summary(query, sentences=3)
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

class GetQuestionTool(Tool):
    name = "get_question"
    description = "Get a question from the GAIA quiz."
    inputs = {"task_id": {"type": "string", "description": "Task ID (optional)", "nullable": True}}
    output_type = "object"

    def forward(self, task_id: str = None) -> dict:
        if task_id:
            response = requests.get(f"{QUESTIONS_ENDPOINT}/{task_id}")
        else:
            response = requests.get(QUESTIONS_ENDPOINT)
        return response.json()

class SubmitAnswerTool(Tool):
    name = "submit_answer"
    description = "Submit an answer to the GAIA quiz."
    inputs = {
        "username": {"type": "string", "description": "Hugging Face username."},
        "code_link": {"type": "string", "description": "Code link (GitHub/GitLab)."},
        "answers": {"type": "object", "description": "List of answers."}
    }
    output_type = "object"

    def forward(self, username: str, code_link: str, answers: list) -> dict:
        payload = {
            "username": username,
            "code_link": code_link,
            "answers": answers
        }
        response = requests.post(SUBMIT_ENDPOINT, json=payload)
        return response.json()

class GAIAgent:
    def __init__(self):
        self.agent = ToolCallingAgent(
            tools=self._get_tools(),
            model=self._get_model()
        )
    
    def _get_model(self):
        # Return a function that takes messages and calls completion
        def model(messages, **kwargs):
            return completion(
                model="groq/llama3-70b-8192",
                api_key=os.getenv("GROQ_API_KEY"),
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
        return model
    
    def _get_tools(self):
        return [
            SearchWebTool(),
            SearchWikipediaTool(),
            GetQuestionTool(),
            SubmitAnswerTool()
        ]
    
    def process_question(self, question: Dict) -> Dict:
        # Create a prompt for the agent
        prompt = f"""
        Question: {question['question']}
        Task ID: {question['task_id']}
        
        Please analyze this question and provide a detailed answer. Use the available tools to gather information if needed.
        """
        
        # Get the agent's response
        response = self.agent.run(prompt)
        
        # Format the answer
        return {
            "task_id": question['task_id'],
            "answer": response
        }

# Initialize the agent
agent = GAIAAgent()

def fetch_questions():
    """Fetch all questions from the GAIA API."""
    try:
        response = requests.get(QUESTIONS_ENDPOINT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def submit_answers(username, code_link, questions):
    """Submit answers to the GAIA API."""
    if not username or not code_link:
        return {"error": "Please provide both username and code link."}
    if not questions or isinstance(questions, dict) and questions.get("error"):
        return {"error": "No questions to answer."}
    answers = []
    for q in questions:
        answer = agent.process_question(q)
        answers.append({"task_id": q["task_id"], "submitted_answer": answer})
    payload = {
        "username": username,
        "code_link": code_link,
        "answers": answers
    }
    try:
        response = requests.post(SUBMIT_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    with gr.Blocks(title="GAIA Quiz Agent") as demo:
        gr.Markdown("# GAIA Quiz Agent")
        with gr.Row():
            username = gr.Textbox(label="Hugging Face Username")
            code_link = gr.Textbox(label="Code Link (GitHub/GitLab)")
        with gr.Row():
            fetch_btn = gr.Button("Fetch Questions")
            submit_btn = gr.Button("Submit Answers")
        questions_output = gr.JSON(label="Questions")
        submission_output = gr.JSON(label="Submission Result")
        fetch_btn.click(fn=fetch_questions, outputs=questions_output)
        submit_btn.click(fn=submit_answers, inputs=[username, code_link, questions_output], outputs=submission_output)
    demo.launch()

if __name__ == "__main__":
    main()
