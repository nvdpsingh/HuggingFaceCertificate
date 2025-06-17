# GAIA Quiz Agent

This project is a Gradio-based agent that interacts with the GAIA quiz API. It uses the Groq API to process questions and submit answers.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nvdpsingh/HuggingFaceCertificate.git
   cd HuggingFaceCertificate
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the project root with your API keys:**
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the App

Run the Gradio app with:
```bash
python app.py
```

Then open your browser and go to:  
**http://127.0.0.1:7860**

## Usage

- Enter your Hugging Face username and code link (GitHub/GitLab repository URL).
- Use the interface to fetch questions and submit answers.
- The agent uses Groq's Llama 3 70B model to process questions and interact with the GAIA quiz API.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 