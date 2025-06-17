"""
tools.py - Tool definitions for the GAIA Quiz Agent
"""
import requests

class WebSearchTool:
    """Tool for searching the web using DuckDuckGo."""
    def run(self, query):
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                return "\n".join([f"{r['title']}: {r['body']}" for r in results])
        except Exception as e:
            return f"Error searching web: {str(e)}"

class WikipediaTool:
    """Tool for searching Wikipedia."""
    def run(self, query):
        try:
            import wikipedia
            return wikipedia.summary(query, sentences=3)
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"

class FileDownloadTool:
    """Tool for downloading files from the GAIA API."""
    BASE_URL = "https://gaia-quiz-api.huggingface.co/files"
    def run(self, task_id):
        try:
            url = f"{self.BASE_URL}/{task_id}"
            response = requests.get(url)
            response.raise_for_status()
            return response.content  # or save to file if needed
        except Exception as e:
            return f"Error downloading file: {str(e)}"

def get_tools():
    """Return a list of all available tools."""
    return [WebSearchTool(), WikipediaTool(), FileDownloadTool()] 