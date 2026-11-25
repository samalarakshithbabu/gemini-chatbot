# Gemini Chatbot Application

A smart chatbot application powered by the **Gemini API (Google Generative AI)** with analytics, research tools, and document processing capabilities.

---

## Problem Statement

Building intelligent conversational agents requires integration with modern LLMs, session management, and extensibility for domain-specific tools. This project demonstrates Gemini API integration with a modular architecture for experimentation and extension.

## Objectives

- Integrate Google Gemini API for intelligent responses
- Provide extensible architecture for domain-specific flows
- Secure API key management via environment variables

## Key Features

- **Interactive Chat Interface** - Real-time communication via Streamlit web UI
- **Gemini API Integration** - Google's Generative AI for intelligent responses
- **Customizable Logic** - Easily extend for domain-specific flows or predefined intents
- **API Key Security** - Environment variable based key management
- **Multiple Entry Points** - `chatbot.py`, `enhanced_chatbot.py`, `nexora_ai.py` for different complexity levels
- **Session History** - Conversation history stored in Streamlit session state (basic)

## Project Structure

```
gemini-chatbot/
├── chatbot.py              # Basic Gemini chatbot
├── enhanced_chatbot.py     # Enhanced version with more features
├── nexora_ai.py           # Advanced implementation (Nexora AI)
├── seed_data.py           # Data seeding utility
├── test_chat.py           # Test suite
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python |
| NLP Engine | Gemini API (Google Generative AI) |
| Frontend | Console I/O (extensible to Flask/React) |
| Communication | Console (can extend to WebSocket/Flask) |
| API Management | Environment variables |

## Installation & Setup

### Prerequisites
- Python 3.8+
- A Gemini API Key (Get it from Google AI Studio)
- Basic Python environment

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/samalarakshithbabu/gemini-chatbot.git
cd gemini-chatbot

# 2. (Optional) Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
# Windows:
set GEMINI_API_KEY=your_api_key_here
# macOS/Linux:
export GEMINI_API_KEY=your_api_key_here

# 5. Run the chatbot
python chatbot.py
```

## How It Works

1. User inputs a message (via console or UI)
2. Chatbot sends prompt to Gemini API with conversation context
3. Gemini model processes and returns intelligent response
4. Chatbot displays response and updates session history

## Example Interaction

```
You: What is the capital of France?
Bot: The capital of France is Paris.

You: Can you explain quantum computing in simple terms?
Bot: Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously...
```

## Potential Applications

- **Customer Support** - Automate FAQs and basic query handling
- **Educational Assistant** - Explain topics, suggest resources
- **Mental Health Chatbot** - Empathetic, supportive responses
- **Interview Bot** - Mock interviews or role-playing
- **Developer Tool** - Code explanations and debugging tips

## Future Improvements

- Voice Interaction (Speech-to-Text + Text-to-Speech)
- Web UI using Flask or React
- Platform Integration: WhatsApp, Telegram, Slack
- User Authentication for personalized chats
- Persistent Memory using database (e.g., MongoDB)
- Document upload and Q&A (RAG implementation)

## Requirements

See `requirements.txt`:
```
google-generativeai
python-dotenv
```

## Configuration

Create `.env` file from `.env.example`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## Author

**SAMALA RAKSHITH BABU**  

**Contact:**  
- Email: samalarakshithbabu7@gmail.com  
- LinkedIn: https://www.linkedin.com/in/samala-rakshith-babu/  
- GitHub: https://github.com/samalarakshithbabu  

---

## License

MIT License