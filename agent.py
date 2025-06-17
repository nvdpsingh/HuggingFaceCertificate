"""
agent.py - Main agent logic for the GAIA Quiz Agent
"""
from tools import get_tools
from litellm import completion
import os

class GAIAAgent:
    """
    GAIAAgent orchestrates tool use and LLM calls to answer GAIA quiz questions.
    """
    def __init__(self, model_name="groq/llama3-70b-8192"):
        self.model_name = model_name
        self.tools = get_tools()
        self.api_key = os.getenv("GROQ_API_KEY")
        # Map tool names to tool instances for easy access
        self.tool_map = {tool.__class__.__name__.lower(): tool for tool in self.tools}

    def answer_question(self, question):
        """
        Use tools based on keywords in the question, then use LLM to answer.
        """
        qtext = question["question"].lower()
        tool_result = None
        tool_used = None
        # Simple keyword-based tool selection
        if "wikipedia" in qtext:
            tool_used = "wikipediatool"
            tool_result = self.tool_map[tool_used].run(question["question"])
        elif "web" in qtext or "search" in qtext:
            tool_used = "websearchtool"
            tool_result = self.tool_map[tool_used].run(question["question"])
        elif "file" in qtext or "download" in qtext:
            tool_used = "filedownloadtool"
            tool_result = self.tool_map[tool_used].run(question["task_id"])
        # Compose prompt for LLM
        if tool_result:
            prompt = f"""
            Question: {question['question']}
            Task ID: {question['task_id']}
            Tool used: {tool_used}
            Tool result: {tool_result}
            Please answer concisely and accurately, using the tool result above if helpful.
            """
        else:
            prompt = f"""
            Question: {question['question']}
            Task ID: {question['task_id']}
            Please answer concisely and accurately.
            """
        response = completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        return response['choices'][0]['message']['content'].strip() 