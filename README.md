Legal chatbot
# AI Legal Chatbot  

## Overview  
This project is a **Flask-based chatbot** that uses **LlamaIndex** and **OpenAI GPT** to retrieve and process legal documents. Users can query a collection of texts via a **web interface**, and the chatbot generates AI-driven responses.  

### Features  
- **AI-powered Q&A** based on document indexing.  
- **Vector-based search** for accurate retrieval of information.  
- **Web-based UI** for user-friendly interaction.  
- **Customizable dataset** for different types of legal texts.  

---

## Project Structure  
📁 chatbot_project/ ├── 📁 templates/ # HTML files for Flask frontend │ └── index.html
├── 📁 static/ # CSS and JavaScript for UI │ ├── styles.css
│ ├── script.js
├── 📁 documents/ # Legal documents for retrieval ├── 📁 tests/ # Unit tests (if implemented) ├── 📄 app.py # Flask backend ├── 📄 utils.py # Utility functions ├── 📄 requirements.txt # Dependencies list ├── 📄 .gitignore # Excluded files ├── 📄 .env.example # Environment variables template ├── 📄 README.md # Documentation


---

## Installation & Setup  
### Clone the repository  
```bash
git clone https://github.com/yourusername/chatbot_project.git
cd chatbot_project

# install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and replace your-api-key-here with your OpenAI API Key

# run the chatbot
python app.py
# Open your browser and go to http://127.0.0.1:5000/ to interact with the chatbot.

