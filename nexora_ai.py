"""
NEXORA AI - Universal Intelligence & Creative Studio
A comprehensive multimodal AI platform
"""
import os
import json
import sqlite3
import hashlib
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

import streamlit as st
from dotenv import load_dotenv
import httpx
from duckduckgo_search import DDGS
from langdetect import detect, LangDetectException
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY", "")
DB_PATH = "nexora_analytics.db"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_database():
    """Initialize all database tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            language TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0
        )
    ''')
    
    # Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            intent TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    
    # Analytics events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            details TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Document uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            filename TEXT,
            file_type TEXT,
            content TEXT,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Media generations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_type TEXT,
            prompt TEXT,
            parameters TEXT,
            result_url TEXT,
            status TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()

# ============================================================================
# GOOGLE GENAI CLIENT
# ============================================================================

class GoogleGenAIClient:
    """
    Client for Google Gemini API using the official google.genai SDK.
    Automatically selects an available working model.
    """
    
    # Default model order - will try each until one works
    MODEL_PREFERENCES = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash-lite",
        "gemini-flash-latest",
        "gemini-3.8-flash",
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None
        self._working_model = None
        self._model_tested = False
    
    def _get_client(self):
        """Get or create the google.genai client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                st.error(f"Failed to create Google GenAI client: {e}")
                return None
        return self._client
    
    def get_available_models(self) -> List[Dict]:
        """Fetch available models from the API."""
        try:
            client = self._get_client()
            if not client:
                return []
            
            models = []
            for m in client.models.list():
                name = m.name if hasattr(m, 'name') else str(m)
                models.append({"name": name})
            return models
        except Exception as e:
            st.error(f"Failed to fetch models: {e}")
            return []
    
    def _find_working_model(self) -> Optional[str]:
        """Test models and find one that works."""
        client = self._get_client()
        if not client:
            return None
        
        for model_name in self.MODEL_PREFERENCES:
            try:
                # Test generation with a simple prompt
                response = client.models.generate_content(
                    model=model_name,
                    contents="Test"
                )
                if response and hasattr(response, 'text') and response.text:
                    return model_name
            except Exception:
                continue
        
        return None
    
    def _get_working_model(self) -> str:
        """Get the working model, testing if necessary."""
        if self._working_model is None and not self._model_tested:
            self._working_model = self._find_working_model()
            self._model_tested = True
        return self._working_model or "gemini-3.5-flash-lite"  # Default fallback
    
    def generate_content(self, prompt: str, model: str = None) -> Tuple[str, bool]:
        """Generate content using Gemini API with automatic model selection and retry."""
        import time
        
        client = self._get_client()
        if not client:
            return "API client not initialized. Check your API key.", False
        
        # Auto-select model if not specified
        if model is None:
            model = self._get_working_model()
        
        # Retry logic for 503 errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                
                if response and hasattr(response, 'text'):
                    text = response.text.strip()
                    if text:
                        return text, True
                    else:
                        return "Empty response from API", False
                        
            except Exception as e:
                error_str = str(e)
                
                # Check for specific errors
                if "503" in error_str or "UNAVAILABLE" in error_str:
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    return "Service temporarily unavailable. Please try again.", False
                
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    # Model not found, try to find another
                    self._working_model = None
                    return f"Model '{model}' not available. Retrying with alternative model.", False
                
                elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    return "API quota exceeded. Please try again later.", False
                
                elif "401" in error_str or "PERMISSION_DENIED" in error_str:
                    return "Invalid API key. Please check your configuration.", False
                
                else:
                    return f"API Error: {error_str[:100]}", False
        
        return "Failed after retries", False

# Initialize client
@st.cache_resource
def get_client():
    return GoogleGenAIClient(API_KEY) if API_KEY else None

client = get_client()

# ============================================================================
# RESEARCH ENGINE
# ============================================================================

def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """Perform web search using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "url": r.get("url", ""), 
                     "snippet": r.get("body", "")} for r in results]
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def deep_research(query: str, mode: str = "detailed") -> Dict:
    """Perform deep research on a topic."""
    search_results = web_search(query, max_results=10 if mode in ["detailed", "deep"] else 5)
    
    # Generate summary using AI
    sources_text = "\n\n".join([f"Sources:\n- {r['title']}: {r['snippet']}" for r in search_results])
    
    research_prompt = f"""
    Research Query: {query}
    
    Search Results:
    {sources_text}
    
    Please provide a comprehensive research report with:
    1. Summary of findings
    2. Key points from each source
    3. Source citations
    4. Any conflicting information
    5. Recommendations for further research
    
    Format in clear markdown.
    """
    
    ai_client = get_client()
    if ai_client:
        report, success = ai_client.generate_content(research_prompt)
    else:
        report = "API key not configured. Unable to generate research report."
        success = False
    
    return {
        "query": query,
        "mode": mode,
        "sources": search_results,
        "report": report,
        "success": success
    }

# ============================================================================
# AI IMAGE GENERATION (Using Pollinations API - Free)
# ============================================================================

def generate_image(prompt: str, width: int = 1024, height: int = 1024, 
                   seed: int = None, style: str = "natural") -> Tuple[str, bool]:
    """Generate image using Pollinations AI (free, no API key needed)."""
    try:
        # Pollinations API - free image generation
        base_url = "https://image.pollinations.ai/prompt/"
        encoded_prompt = httpx.utils.quote(prompt)
        
        params = f"?width={width}&height={height}&nologo=true"
        if seed:
            params += f"&seed={seed}"
        if style == "anime":
            params += "&style=anime"
        elif style == "cartoon":
            params += "&style=cartoon"
        elif style == "oil painting":
            params += "&style=oilpainting"
        elif style == "photorealistic":
            params += "&seed=42"
        
        image_url = f"{base_url}{encoded_prompt}{params}"
        return image_url, True
    except Exception as e:
        return str(e), False

# ============================================================================
# AI VIDEO GENERATION (Simulated - Requires Replicate/Runway API)
# ============================================================================

def generate_video(prompt: str, duration: int = 5, aspect_ratio: str = "16:9") -> Tuple[Dict, bool]:
    """Generate video using external API. Returns status and info."""
    # Note: Actual video generation requires APIs like RunwayML, Replicate, or Luma Dream Machine
    # This implementation provides the structure and would integrate with such APIs
    
    # Check if Replicate API token is available
    replicate_token = os.getenv("REPLICATE_API_TOKEN", "")
    
    if not replicate_token:
        return {
            "status": "requires_api",
            "message": "Video generation requires Replicate API token",
            "setup_url": "https://replicate.com/account/api-tokens",
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio
        }, False
    
    try:
        # This would be the actual API call for Replicate
        # Using Model: zootimba/animatediff-cli (example)
        # import replicate
        # output = replicate.run(
        #     "model-name:version",
        #     input={"prompt": prompt, "num_frames": duration * 8}
        # )
        return {"status": "pending", "video_url": "", "message": "API integration needed"}, False
    except Exception as e:
        return {"status": "error", "message": str(e)}, False

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_document(file) -> Tuple[str, bool]:
    """Process uploaded document and extract text."""
    try:
        file_type = file.name.split('.')[-1].lower()
        
        if file.size > MAX_FILE_SIZE:
            return "File too large. Maximum size is 10MB.", False
        
        if file_type == 'pdf':
            pdf = PdfReader(file)
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            return text, True
        
        elif file_type in ['docx', 'doc']:
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text, True
        
        elif file_type == 'txt':
            return file.read().decode('utf-8'), True
        
        elif file_type in ['csv', 'xlsx', 'xls']:
            if file_type == 'csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df.to_string(), True
        
        else:
            return "Unsupported file format", False
            
    except Exception as e:
        return str(e), False

# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

def create_conversation(title: str = "New Chat", language: str = "en") -> str:
    """Create a new conversation."""
    conv_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:12]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO conversations (id, title, language) VALUES (?, ?, ?)",
        (conv_id, title[:50], language)
    )
    conn.commit()
    conn.close()
    return conv_id

def save_message(conv_id: str, role: str, content: str, intent: str = ""):
    """Save a message to the conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, intent) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, intent)
    )
    
    cursor.execute(
        "UPDATE conversations SET message_count = message_count + 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conv_id,)
    )
    conn.commit()
    conn.close()

def get_conversation_history(conv_id: str) -> List[Dict]:
    """Get conversation messages."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp",
        (conv_id,)
    )
    messages = [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in cursor.fetchall()]
    conn.close()
    return messages

def get_all_conversations() -> List[Dict]:
    """Get all conversations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, title, language, created_at, updated_at, message_count FROM conversations ORDER BY updated_at DESC"
    )
    conversations = [{"id": r[0], "title": r[1], "language": r[2], 
                      "created_at": r[3], "updated_at": r[4], "message_count": r[5]} 
                     for r in cursor.fetchall()]
    conn.close()
    return conversations

def delete_conversation(conv_id: str):
    """Delete a conversation and its messages."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()

def update_conversation_title(conv_id: str, title: str):
    """Update conversation title."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title[:50], conv_id))
    conn.commit()
    conn.close()

# ============================================================================
# ANALYTICS
# ============================================================================

def log_event(event_type: str, details: Dict = None):
    """Log an analytics event."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO analytics_events (event_type, details) VALUES (?, ?)",
        (event_type, json.dumps(details) if details else "{}")
    )
    conn.commit()
    conn.close()

def get_analytics() -> Dict:
    """Get analytics data."""
    conn = sqlite3.connect(DB_PATH)
    
    # Basic stats
    stats = pd.read_sql_query('''
        SELECT COUNT(*) as total_conversations FROM conversations
    ''', conn)
    
    message_count = pd.read_sql_query('''
        SELECT COUNT(*) as total_messages FROM messages
    ''', conn)
    
    # Media generations
    media = pd.read_sql_query('''
        SELECT generation_type, COUNT(*) as count FROM media_generations GROUP BY generation_type
    ''', conn)
    
    # Feature usage
    events = pd.read_sql_query('''
        SELECT event_type, COUNT(*) as count FROM analytics_events GROUP BY event_type
    ''', conn)
    
    conn.close()
    
    return {
        "conversations": stats.iloc[0]['total_conversations'] if len(stats) > 0 else 0,
        "messages": message_count.iloc[0]['total_messages'] if len(message_count) > 0 else 0,
        "media_generations": media,
        "events": events
    }

# ============================================================================
# LANGUAGE UTILITIES
# ============================================================================

def detect_language(text: str) -> str:
    """Detect language of input text."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate text to target language using Gemini."""
    if not client:
        return text
    
    prompt = f"""
    Translate the following text to {target_lang}. 
    Preserve the meaning, tone, and formatting.
    
    Text: {text}
    
    Translation:
    """
    
    result, success = client.generate_content(prompt)
    return result if success else text

# ============================================================================
# NEXORA AI - MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="NEXORA AI - Universal Intelligence",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS - Enhanced Visibility & Accessibility
    st.markdown("""
    <style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1a 100%);
    }
    
    /* Main Heading */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar Section */
    .sidebar-section {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #3D3D5C;
    }
    .sidebar-section p,
    .sidebar-section li,
    .sidebar-section strong {
        color: #FFFFFF !important;
    }
    
    /* Sidebar headings */
    .sidebar-section h1,
    .sidebar-section h2,
    .sidebar-section h3,
    .sidebar-section h4 {
        color: #FFFFFF !important;
    }
    
    /* Sidebar buttons for conversations */
    .stSidebar button[data-testid="baseButton-secondary"] {
        background-color: #1E1E2E !important;
        color: #FFFFFF !important;
        border: 1px solid #3D3D5C !important;
        border-radius: 8px !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    .stSidebar button[data-testid="baseButton-secondary"]:hover {
        background-color: #2D2D3E !important;
        border-color: #5D5D7C !important;
    }
    
    /* Sidebar delete button */
    .stSidebar button[data-testid="baseButton-secondary"]:has(svg) {
        background-color: #2D2D3E !important;
        border: 1px solid #3D3D5C !important;
    }
    
    /* Sidebar navigation buttons */
    .stSidebar [data-testid="stSidebarNav"] button {
        background-color: #1E1E2E !important;
        color: #FFFFFF !important;
        border: 1px solid #3D3D5C !important;
    }
    .stSidebar [data-testid="stSidebarNav"] button:hover {
        background-color: #2D2D3E !important;
    }
    
    /* Sidebar - Active navigation */
    .stSidebar [data-testid="stSidebarNav"] button[data-testid="baseButton-primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    /* Chat Messages - User (Light Blue Background) */
    .chat-message.user-message {
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 12px;
        background: #DBEAFE;
        border: 1px solid #93C5FD;
    }
    .chat-message.user-message strong {
        color: #172033;
    }
    .chat-message.user-message div[style*="color"] {
        color: #172033 !important;
    }
    
    /* Chat Messages - Bot (White Background) */
    .chat-message.bot-message {
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 12px;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
    }
    .chat-message.bot-message strong {
        color: #172033;
    }
    .chat-message.bot-message div[style*="color"] {
        color: #172033 !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* All Input Fields - White Background with Dark Text */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within,
    .stMultiSelect > div > div > div:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }
    
    /* Input Placeholder */
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #64748B !important;
        opacity: 1 !important;
    }
    
    /* Text Area Specific */
    .stTextArea > div > div {
        background-color: #FFFFFF !important;
        color: #172033 !important;
    }
    
    /* Primary Button - Blue */
    .stButton > button[kind="primary"],
    div.stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button[kind="primary"]:hover,
    div.stButton > button:hover {
        background-color: #1D4ED8 !important;
    }
    
    /* Secondary Button - Light Grey */
    .stButton > button[kind="secondary"],
    button[kind="secondary"] {
        background-color: #F1F5F9 !important;
        color: #172033 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stButton > button[kind="secondary"]:hover,
    button[kind="secondary"]:hover {
        background-color: #E2E8F0 !important;
    }
    
    /* All Headings - White */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Labels and Descriptions - Light Grey */
    .stMarkdown p,
    label,
    .stCaption,
    div[data-testid="stMarkdownContainer"] p {
        color: #E2E8F0 !important;
    }
    
    /* Success Messages - Dark Green on Light Green */
    div[data-testid="stSuccess"] {
        background-color: #DCFCE7 !important;
        color: #166534 !important;
        border: 1px solid #86EFAC !important;
        border-radius: 8px !important;
    }
    
    /* Error Messages - Dark Red on Light Red */
    div[data-testid="stError"] {
        background-color: #FEE2E2 !important;
        color: #991B1B !important;
        border: 1px solid #FCA5A5 !important;
        border-radius: 8px !important;
    }
    
    /* Warning Messages */
    div[data-testid="stWarning"] {
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #FCD34D !important;
        border-radius: 8px !important;
    }
    
    /* Info Messages */
    div[data-testid="stInfo"] {
        background-color: #E0F2FE !important;
        color: #075985 !important;
        border: 1px solid #7DD3FC !important;
        border-radius: 8px !important;
    }
    
    /* Selectbox Dropdown */
    .stSelectbox > div > div > div {
        background-color: #FFFFFF !important;
        color: #172033 !important;
        border: 1px solid #CBD5E1 !important;
    }
    
    /* Sidebar Text */
    .stSidebar {
        background-color: #1E1E2E !important;
        color: #FFFFFF !important;
    }
    .stSidebar .stMarkdown {
        color: #FFFFFF !important;
    }
    .stSidebar p,
    .stSidebar li,
    .stSidebar label {
        color: #FFFFFF !important;
    }
    
    /* Sidebar Buttons - All Buttons */
    .stSidebar .stButton > button {
        background-color: #2D2D3E !important;
        color: #FFFFFF !important;
        border: 1px solid #3D3D5C !important;
        border-radius: 8px !important;
        text-align: left !important;
    }
    .stSidebar .stButton > button:hover {
        background-color: #3D3D4E !important;
        border-color: #5D5D7C !important;
    }
    
    /* Sidebar - Active Button (Primary) */
    .stSidebar .stButton > button[data-testid="baseButton-primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    /* Sidebar - Secondary Button */
    .stSidebar .stButton > button[data-testid="baseButton-secondary"] {
        background-color: #2D2D3E !important;
        color: #FFFFFF !important;
        border: 1px solid #3D3D5C !important;
    }
    
    /* Sidebar Headers */
    .stSidebar h1,
    .stSidebar h2,
    .stSidebar h3,
    .stSidebar h4 {
        color: #FFFFFF !important;
    }
    
    /* Sidebar divider */
    .stSidebar hr {
        border-color: #3D3D5C !important;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background-color: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 8px !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background-color: #2563EB !important;
    }
    
    /* Spinner */
    div[data-testid="stSpinner"] {
        color: #2563EB !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255,255,255,0.1) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #E2E8F0 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.08) !important;
        color: #E2E8F0 !important;
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        color: #2563EB !important;
    }
    
    /* Radio Buttons */
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #E2E8F0 !important;
    }
    
    /* DataFrame/Table */
    .stDataFrame {
        background-color: #FFFFFF !important;
    }
    
    /* Divider */
    hr {
        border-color: #CBD5E1 !important;
    }
    
    /* Cursor */
    textarea, input {
        caret-color: #2563EB !important;
    }
    
    /* Ensure white text for metrics */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] {
        color: #E2E8F0 !important;
    }
    
    /* Chat input area */
    .stChatInput {
        background-color: transparent !important;
    }
    .stChatInput > div {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    .stChatInput input {
        color: #172033 !important;
        background-color: #FFFFFF !important;
    }
    .stChatInput input::placeholder {
        color: #64748B !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'page' not in st.session_state:
        st.session_state.page = 'chat'
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>🚀 NEXORA AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Universal Intelligence & Creative Studio</p>", unsafe_allow_html=True)
        st.divider()
        
        # New Chat Button
        if st.button("✨ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_conversation = create_conversation()
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        st.markdown("**📁 Conversations**")
        
        # Conversation list
        conversations = get_all_conversations()
        for conv in conversations[:10]:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(conv['title'][:25] + "..." if len(conv['title']) > 25 else conv['title'], 
                           key=f"conv_{conv['id']}", help=conv['title']):
                    st.session_state.current_conversation = conv['id']
                    st.session_state.messages = get_conversation_history(conv['id'])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}"):
                    delete_conversation(conv['id'])
                    if st.session_state.current_conversation == conv['id']:
                        st.session_state.current_conversation = None
                        st.session_state.messages = []
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        nav_options = [
            ("💬 AI Chat", "chat"),
            ("🔬 Deep Research", "research"),
            ("🎬 Video Studio", "video"),
            ("🎨 Image Studio", "image"),
            ("📄 Document Assistant", "document"),
            ("📊 Analytics", "analytics"),
            ("⚙️ Settings", "settings"),
        ]
        
        for label, page in nav_options:
            if st.button(label, use_container_width=True, 
                        type="secondary" if st.session_state.page != page else "primary"):
                st.session_state.page = page
                log_event("navigation", {"from": st.session_state.page, "to": page})
                st.rerun()
        
        st.divider()
        
        # API Status
        if API_KEY:
            st.success("🟢 API Key Configured")
        else:
            st.error("🔴 No API Key - Add to .env file")
    
    # Main Content Area
    if st.session_state.page == 'chat':
        render_chat_page()
    elif st.session_state.page == 'research':
        render_research_page()
    elif st.session_state.page == 'video':
        render_video_page()
    elif st.session_state.page == 'image':
        render_image_page()
    elif st.session_state.page == 'document':
        render_document_page()
    elif st.session_state.page == 'analytics':
        render_analytics_page()
    elif st.session_state.page == 'settings':
        render_settings_page()

# ============================================================================
# PAGE: AI CHAT
# ============================================================================

def render_chat_page():
    """Render the main chat interface."""
    
    # Welcome screen if no conversation
    if not st.session_state.current_conversation:
        st.markdown("""
        <div style='text-align: center; padding: 60px 20px;'>
            <h1 style='font-size: 3rem; margin-bottom: 20px; color: #FFFFFF !important;'>🤖 Welcome to NEXORA AI</h1>
            <p style='font-size: 1.2rem; color: #E2E8F0; max-width: 600px; margin: 0 auto;'>
                One platform that understands, researches, creates, explains and solves.
            </p>
            <div style='margin-top: 40px;'>
                <p style='color: #E2E8F0;'>Try asking me:</p>
                <p style='color: #93C5FD;'>"Explain quantum computing in simple terms"</p>
                <p style='color: #C4B5FD;'>"Write a Python function to sort data"</p>
                <p style='color: #6EE7B7;'>"What are the benefits of renewable energy?"</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Start a New Conversation", type="primary"):
            st.session_state.current_conversation = create_conversation()
            st.session_state.messages = []
            st.rerun()
        return
    
    # Chat interface
    st.markdown(f"**💬 Conversation** | ID: `{st.session_state.current_conversation}`")
    
    # Rename conversation
    col1, col2 = st.columns([4, 1])
    with col1:
        conv_title = st.text_input("Title", value="New Chat", key="conv_title")
    with col2:
        if st.button("Save Title"):
            update_conversation_title(st.session_state.current_conversation, conv_title)
            st.success("Saved!")
    
    # Chat history display
    st.markdown("---")
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class='chat-message user-message' style='background: #DBEAFE; border: 1px solid #93C5FD;'>
                <strong style='color: #172033;'>You:</strong><br>
                <span style='color: #172033;'>{msg['content']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='chat-message bot-message' style='background: #FFFFFF; border: 1px solid #CBD5E1;'>
                <strong style='color: #172033;'>🤖 NEXORA:</strong><br>
                <span style='color: #172033;'>{msg['content']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    # Input options - Language selector, Web Search toggle and Send
    col1, col2, col3 = st.columns([4, 2, 2])
    with col1:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 8px;'>Ask anything...</p>", unsafe_allow_html=True)
        user_input = st.text_area("", value="", height=80, placeholder="Type your question here...", label_visibility="collapsed")
    with col2:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 8px;'>Language</p>", unsafe_allow_html=True)
        target_lang = st.selectbox("", ["en", "hi", "te", "es", "fr", "de", "zh", "ja"], label_visibility="collapsed")
    with col3:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 8px;'>Options</p>", unsafe_allow_html=True)
        search_mode = st.checkbox("Web Search", value=False, help="Search the web for current information")
        submit = st.button("Send", type="primary", use_container_width=True)
    
    if submit and user_input.strip():
            # Detect language
            detected_lang = detect_language(user_input)
            
            # Save user message
            save_message(st.session_state.current_conversation, 'user', user_input)
            
            # Prepare prompt
            context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])
            
            prompt = f"""
            You are NEXORA AI, a helpful AI assistant.
            
            Context of conversation:
            {context}
            
            Current user question: {user_input}
            
            {'Also search the web for current information.' if search_mode else ''}
            
            Please provide a clear, helpful, and accurate response.
            """
            
            # Add translation instruction if needed
            if target_lang != detected_lang:
                prompt += f"\n\nPlease respond in {target_lang} language."
            
            # Generate response
            with st.spinner("🤔 NEXORA is thinking..."):
                # Get the proper client
                ai_client = get_client()
                if ai_client:
                    response, success = ai_client.generate_content(prompt)
                    if not success:
                        st.error(response)
                        return
                else:
                    response = "❌ API key not configured. Please add your Google API key to the .env file."
                    success = False
                
                # Save bot response
                save_message(st.session_state.current_conversation, 'bot', response)
                st.session_state.messages.append({'role': 'user', 'content': user_input})
                st.session_state.messages.append({'role': 'bot', 'content': response})
                log_event("chat_message", {"success": success, "has_web_search": search_mode})
            
            st.rerun()

# ============================================================================
# PAGE: DEEP RESEARCH
# ============================================================================

def render_research_page():
    """Render the deep research interface."""
    st.markdown("<h1>🔬 Deep Research Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0;'>Research any topic with AI-powered search and analysis.</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #E2E8F0; margin-bottom: 8px;'>Research Mode</p>", unsafe_allow_html=True)
    research_mode = st.selectbox("", 
                                  ["Quick Answer", "Detailed Answer", "Deep Research", 
                                   "Comparative Analysis", "Step-by-Step Explanation"],
                                  label_visibility="collapsed")
    
    st.markdown("<p style='color: #E2E8F0; margin-bottom: 8px;'>Research Topic</p>", unsafe_allow_html=True)
    query = st.text_area("", 
                          placeholder="Enter your research question...",
                          height=120,
                          label_visibility="collapsed")
    
    if st.button("🔍 Start Research", type="primary"):
        if query.strip():
            with st.spinner("🔍 Searching and analyzing..."):
                results = deep_research(query, mode=research_mode.lower().replace(" ", "_"))
            
            if results.get("success"):
                st.markdown("---")
                st.markdown("### 📋 Research Report")
                st.markdown(results["report"], unsafe_allow_html=True)
                
                st.markdown("### 📚 Sources")
                for i, source in enumerate(results["sources"], 1):
                    st.markdown(f"""
                    **{i}. [{source['title']}]({source['url']})**
                    {source['snippet'][:200]}...
                    """)
                log_event("research", {"mode": research_mode, "sources_found": len(results["sources"])})
            else:
                st.error(results.get("report", "Research failed"))

# ============================================================================
# PAGE: VIDEO STUDIO
# ============================================================================

def render_video_page():
    """Render the video generation interface."""
    st.markdown("<h1>🎬 AI Video Studio</h1>", unsafe_allow_html=True)
    st.markdown("Generate videos from text prompts using AI.")
    
    st.markdown("<p style='color: #E2E8F0;'>Describe your video</p>", unsafe_allow_html=True)
    prompt = st.text_area("", 
                          placeholder="A cinematic shot of a futuristic city at night with flying cars...",
                          height=100,
                          label_visibility="collapsed")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Aspect Ratio</p>", unsafe_allow_html=True)
        aspect_ratio = st.selectbox("", ["16:9", "9:16", "1:1", "4:3"], label_visibility="collapsed")
    with col2:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Duration (seconds)</p>", unsafe_allow_html=True)
        duration = st.slider("", 3, 10, 5, label_visibility="collapsed")
    with col3:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Style</p>", unsafe_allow_html=True)
        style = st.selectbox("", ["cinematic", "documentary", "animation", "realistic"], label_visibility="collapsed")
    
    if st.button("🎬 Generate Video", type="primary"):
        if prompt.strip():
            with st.spinner("🎬 Generating video... This may take a few minutes."):
                result, success = generate_video(prompt, duration, aspect_ratio)
            
            if success:
                st.video(result.get("video_url"))
                st.success("Video generated successfully!")
                log_event("video_generation", {"success": True})
            else:
                if result.get("status") == "requires_api":
                    st.warning("""
                    ⚠️ **Video Generation Requires API Setup**
                    
                    To enable video generation, you need to configure an external API like Replicate.
                    
                    **Setup Instructions:**
                    1. Get an API token from [Replicate](https://replicate.com/account/api-tokens)
                    2. Add to your .env file: `REPLICATE_API_TOKEN=your_token_here`
                    3. Restart the application
                    
                    **Note:** Video generation may incur costs through the API provider.
                    """)
                else:
                    st.error(f"Generation failed: {result.get('message', 'Unknown error')}")

# ============================================================================
# PAGE: IMAGE STUDIO
# ============================================================================

def render_image_page():
    """Render the image generation interface."""
    st.markdown("<h1>🎨 AI Image Studio</h1>", unsafe_allow_html=True)
    st.markdown("Generate images from text using Pollinations AI (free!).")
    
    st.markdown("<p style='color: #E2E8F0;'>Describe the image you want</p>", unsafe_allow_html=True)
    prompt = st.text_area("", 
                          placeholder="A beautiful sunset over mountains...",
                          height=100,
                          label_visibility="collapsed")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Width (px)</p>", unsafe_allow_html=True)
        width = st.number_input("", 256, 2048, 1024, label_visibility="collapsed")
    with col2:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Height (px)</p>", unsafe_allow_html=True)
        height = st.number_input("", 256, 2048, 1024, label_visibility="collapsed")
    with col3:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Style</p>", unsafe_allow_html=True)
        style = st.selectbox("", ["natural", "anime", "cartoon", "oil painting", "photorealistic"], label_visibility="collapsed")
    
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Seed (optional)</p>", unsafe_allow_html=True)
        seed = st.number_input("", min_value=0, value=0, step=1, label_visibility="collapsed")
    with col5:
        st.markdown("<p style='color: #E2E8F0; margin-bottom: 4px;'>Options</p>", unsafe_allow_html=True)
        enhance_prompt = st.checkbox("Enhance Prompt", value=True, help="Improve your prompt using AI")
    
    if st.button("🎨 Generate Image", type="primary"):
        if prompt.strip():
            with st.spinner("🎨 Generating image..."):
                final_prompt = prompt
                if enhance_prompt:
                    ai_client = get_client()
                    if ai_client:
                        enhance_result, _ = ai_client.generate_content(
                            f"Enhance this image prompt to be more detailed and artistic: {prompt}"
                        )
                        final_prompt = enhance_result if enhance_result else prompt
                
                image_url, success = generate_image(final_prompt, width, height, 
                                                   seed if seed > 0 else None, style)
            
            if success:
                st.markdown("### Generated Image")
                st.image(image_url, use_container_width=True)
                
                col_dl, _ = st.columns([1, 4])
                with col_dl:
                    st.markdown(f"[📥 Download]({image_url})")
                
                log_event("image_generation", {"success": True, "style": style})
            else:
                st.error("Failed to generate image")

# ============================================================================
# PAGE: DOCUMENT ASSISTANT
# ============================================================================

def render_document_page():
    """Render the document assistant interface."""
    st.markdown("<h1>📄 Document Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0;'>Upload documents and get AI-powered insights.</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #E2E8F0; margin-bottom: 8px;'>Upload document</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", 
                                      type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
                                      label_visibility="collapsed",
                                      help="Supported formats: PDF, DOCX, TXT, CSV, Excel")
    
    if uploaded_file:
        with st.spinner("📄 Processing document..."):
            content, success = process_document(uploaded_file)
        
        if success:
            st.success(f"✅ Document loaded: {uploaded_file.name}")
            
            # Store document in session state
            if 'current_document' not in st.session_state:
                st.session_state.current_document = content
            
            tab1, tab2 = st.tabs(["📖 Document View", "❓ Ask Questions"])
            
            with tab1:
                st.text_area("Document Content", value=content[:10000], height=400, label_visibility="visible")
            
            with tab2:
                st.markdown("<p style='color: #E2E8F0; margin-bottom: 8px;'>Ask about the document</p>", unsafe_allow_html=True)
                question = st.text_input("", placeholder="What is this document about?", label_visibility="collapsed")
                
                if st.button("Get Answer"):
                    with st.spinner("🤔 Analyzing..."):
                        if client:
                            prompt = f"""
                            Document content:
                            {content[:5000]}
                            
                            Question: {question}
                            
                            Answer based on the document above:
                            """
                            answer, success = client.generate_content(prompt)
                            if success:
                                st.markdown("### Answer")
                                st.markdown(answer)
                                log_event("document_query", {"filename": uploaded_file.name})
                            else:
                                st.error(f"Error: {answer}")
                        else:
                            st.error("API key not configured")
            
            # Delete document button
            if st.button("Clear Document"):
                if 'current_document' in st.session_state:
                    del st.session_state.current_document
                st.rerun()
        else:
            st.error(f"Failed to process document: {content}")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

def render_analytics_page():
    """Render the analytics dashboard."""
    st.markdown("<h1>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0;'>Track your NEXORA AI usage and performance.</p>", unsafe_allow_html=True)
    
    analytics = get_analytics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Conversations", analytics['conversations'])
    with col2:
        st.metric("Total Messages", analytics['messages'])
    with col3:
        st.metric("Images Generated", analytics['media_generations'][
            analytics['media_generations']['generation_type'] == 'image'
        ]['count'].sum() if len(analytics['media_generations']) > 0 else 0)
    with col4:
        st.metric("Videos Generated", analytics['media_generations'][
            analytics['media_generations']['generation_type'] == 'video'
        ]['count'].sum() if len(analytics['media_generations']) > 0 else 0)
    
    # Recent conversations
    st.markdown("### 📁 Recent Conversations")
    conversations = get_all_conversations()
    if conversations:
        df = pd.DataFrame(conversations[:10])
        st.dataframe(df[['title', 'message_count', 'updated_at']], use_container_width=True)
    else:
        st.info("No conversations yet. Start chatting to see analytics!")

# ============================================================================
# PAGE: SETTINGS
# ============================================================================

def render_settings_page():
    """Render settings page."""
    st.markdown("<h1>⚙️ Settings</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E2E8F0;'>Configure your NEXORA AI application</p>", unsafe_allow_html=True)
    
    st.markdown("### 🔑 API Configuration")
    st.info(f"Current API Key: {'✅ Configured' if API_KEY else '❌ Not configured'}")
    
    with st.expander("📝 Configuration Guide"):
        st.markdown("""
        **Google API Key:**
        1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Create a new API key
        3. Add to `.env` file: `GOOGLE_API_KEY=your_key_here`
        
        **Replicate API Token (for Video):**
        1. Go to [Replicate](https://replicate.com/account/api-tokens)
        2. Create API token
        3. Add: `REPLICATE_API_TOKEN=your_token`
        """)
    
    st.markdown("### 📋 Model Information")
    
    ai_client = get_client()
    if ai_client:
        models = ai_client.get_available_models()
        if models:
            st.write("**Available Models:**")
            for m in models[:5]:
                st.write(f"- {m['name']}: {m.get('description', 'N/A')[:100]}")
        else:
            st.warning("Could not fetch models. Check your API key.")
    else:
        st.error("Configure API key to see available models.")
    
    st.markdown("### 🗑️ Data Management")
    if st.button("Clear All Conversations"):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        cursor.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()
        st.success("All conversations cleared!")
        st.rerun()

if __name__ == "__main__":
    main()