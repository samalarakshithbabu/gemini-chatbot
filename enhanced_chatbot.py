"""
Enhanced Chatbot with Analytics, Logging, and Evaluation
Features:
- Conversation logging to SQLite
- Intent classification with LLM
- Analytics dashboard
- Intent evaluation with confusion matrix
"""
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
import sqlite3
import json
from datetime import datetime
from typing import List, Tuple, Dict
import pandas as pd
import plotly.express as px
from collections import Counter
import hashlib

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key is missing. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

# Database setup
DB_PATH = "chatbot_analytics.db"

def init_db():
    """Initialize database tables for conversation logging and analytics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_input TEXT,
            bot_response TEXT,
            intent TEXT,
            intent_confidence REAL,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Intents table for training/evaluation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS intent_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            intent TEXT NOT NULL,
            is_test BOOLEAN DEFAULT 0
        )
    ''')
    
    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            rating INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    
    # Unanswered questions tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS unanswered_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            count INTEGER DEFAULT 1,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Intent definitions for classification
INTENT_PATTERNS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "howdy"],
    "product_info": ["what is", "tell me about", "product", "feature", "how does work"],
    "pricing": ["price", "cost", "how much", "pricing", "fee", "charge"],
    "support": ["help", "support", "issue", "problem", "error", "not working"],
    "faq": ["question", "faq", "frequently asked", "common"],
    "feedback": ["feedback", "suggestion", "improve", "opinion"],
    "goodbye": ["bye", "goodbye", "see you", "thanks", "thank you"],
    "fallback": []  # Unknown intents
}

def classify_intent(text: str) -> Tuple[str, float]:
    """
    Classify user intent using keyword matching + LLM fallback.
    Returns (intent, confidence_score).
    """
    text_lower = text.lower()
    
    # Keyword-based classification
    intent_scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if pattern in text_lower:
                score += 1
        intent_scores[intent] = score
    
    # Find best matching intent
    if max(intent_scores.values()) > 0:
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent] / len(text.split())
        return (best_intent, min(1.0, confidence * 2))
    
    # Fallback to LLM classification for unknown intents
    try:
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""
        Classify this user query into one of these intents: greeting, product_info, pricing, support, faq, feedback, goodbye.
        Query: "{text}"
        Return only the intent name, nothing else.
        """
        response = model.generate_content(prompt)
        intent = response.text.strip().lower()
        
        # Validate intent
        if intent in INTENT_PATTERNS:
            return (intent, 0.7)
        else:
            return ("fallback", 0.5)
    except Exception:
        return ("fallback", 0.5)

def log_conversation(session_id: str, user_input: str, bot_response: str, 
                     intent: str, intent_confidence: float, response_time: float):
    """Log conversation to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO conversations (session_id, user_input, bot_response, 
                                   intent, intent_confidence, response_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (session_id, user_input, bot_response, intent, intent_confidence, response_time))
    
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Track unanswered questions (fallback intents)
    if intent == "fallback":
        track_unanswered(user_input)
    
    return conversation_id

def track_unanswered(question: str):
    """Track questions that weren't well-classified."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO unanswered_questions (question, count) VALUES (?, 1)
        ON CONFLICT(question) DO UPDATE SET count = count + 1, 
        last_seen = CURRENT_TIMESTAMP
    ''', (question.lower().strip(),))
    
    conn.commit()
    conn.close()

def save_feedback(conversation_id: int, rating: int, comment: str = ""):
    """Save user feedback for a conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO feedback (conversation_id, rating, comment) VALUES (?, ?, ?)
    ''', (conversation_id, rating, comment))
    
    conn.commit()
    conn.close()

def add_intent_example(text: str, intent: str, is_test: bool = False):
    """Add labeled example for intent training."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO intent_examples (text, intent, is_test) VALUES (?, ?, ?)
    ''', (text, intent, is_test))
    
    conn.commit()
    conn.close()

def get_analytics() -> Dict:
    """Get analytics data from database."""
    conn = sqlite3.connect(DB_PATH)
    
    # Conversation stats
    conv_stats = pd.read_sql_query('''
        SELECT 
            COUNT(*) as total_conversations,
            COUNT(DISTINCT session_id) as unique_users,
            AVG(response_time) as avg_response_time
        FROM conversations
    ''', conn)
    
    # Intent distribution
    intent_dist = pd.read_sql_query('''
        SELECT intent, COUNT(*) as count
        FROM conversations
        GROUP BY intent
        ORDER BY count DESC
    ''', conn)
    
    # Fallback rate
    fallback_rate = pd.read_sql_query('''
        SELECT 
            CAST(COUNT(CASE WHEN intent = 'fallback' THEN 1 END) AS FLOAT) / 
            COUNT(*) * 100 as fallback_percentage
        FROM conversations
    ''', conn)
    
    # Satisfaction from feedback
    satisfaction = pd.read_sql_query('''
        SELECT AVG(rating) as avg_rating, COUNT(*) as total_feedback
        FROM feedback
    ''', conn)
    
    # Unanswered questions
    unanswered = pd.read_sql_query('''
        SELECT question, count
        FROM unanswered_questions
        ORDER BY count DESC
        LIMIT 10
    ''', conn)
    
    # Time series - conversations per day
    timeline = pd.read_sql_query('''
        SELECT DATE(timestamp) as date, COUNT(*) as conversations
        FROM conversations
        GROUP BY DATE(timestamp)
        ORDER BY date
    ''', conn)
    
    conn.close()
    
    return {
        "conv_stats": conv_stats,
        "intent_dist": intent_dist,
        "fallback_rate": fallback_rate,
        "satisfaction": satisfaction,
        "unanswered": unanswered,
        "timeline": timeline
    }

def evaluate_intent_classification() -> Dict:
    """
    Evaluate intent classification on held-out test set.
    Returns confusion matrix and metrics.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Get test examples
    test_examples = pd.read_sql_query('''
        SELECT text, intent FROM intent_examples WHERE is_test = 1
    ''', conn)
    
    conn.close()
    
    if len(test_examples) == 0:
        return {"error": "No test examples available. Add examples with is_test=1."}
    
    # Evaluate on test set
    y_true = []
    y_pred = []
    
    for _, row in test_examples.iterrows():
        predicted_intent, _ = classify_intent(row['text'])
        y_true.append(row['intent'])
        y_pred.append(predicted_intent)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    labels = list(INTENT_PATTERNS.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "labels": labels,
        "test_size": len(test_examples)
    }

# Session ID management
def get_session_id() -> str:
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = hashlib.md5(
            f"{datetime.now()}".encode()
        ).hexdigest()[:8]
    return st.session_state['session_id']

# Streamlit App
def main():
    st.set_page_config(page_title="Enhanced Q&A Bot", layout="wide")
    st.header("🤖 Enhanced Q&A Bot with Analytics")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics Dashboard", "🧪 Intent Evaluation"])
    
    with tab1:
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        if 'conversation_start_time' not in st.session_state:
            st.session_state['conversation_start_time'] = datetime.now()
        
        # Chat interface
        user_input = st.text_input("Ask a question:", key="chat_input")
        submit = st.button("Ask", key="chat_submit")
        
        if submit and user_input:
            start_time = datetime.now()
            
            # Classify intent
            intent, confidence = classify_intent(user_input)
            
            # Get response from Gemini
            try:
                model = genai.GenerativeModel("gemini-pro")
                chat = model.start_chat(history=[])
                response = chat.send_message(user_input, stream=True)
                
                response_text = ""
                for chunk in response:
                    response_text += chunk.text
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Log conversation
                session_id = get_session_id()
                conv_id = log_conversation(
                    session_id, user_input, response_text, 
                    intent, confidence, response_time
                )
                
                # Add to session chat history
                st.session_state['chat_history'].append({
                    "role": "You",
                    "text": user_input,
                    "intent": intent,
                    "confidence": confidence,
                    "conv_id": conv_id
                })
                st.session_state['chat_history'].append({
                    "role": "Bot",
                    "text": response_text
                })
                
                st.session_state['last_conv_id'] = conv_id
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Display chat history
        st.subheader("Chat History")
        for i, msg in enumerate(st.session_state['chat_history']):
            if msg["role"] == "You":
                st.write(f"**You:** {msg['text']}")
                st.caption(f"Intent: {msg.get('intent', 'N/A')} (confidence: {msg.get('confidence', 0):.2f})")
            else:
                st.write(f"**Bot:** {msg['text']}")
            st.divider()
        
        # Feedback section
        if 'last_conv_id' in st.session_state:
            st.subheader("Rate this response")
            col1, col2 = st.columns([3, 1])
            with col1:
                comment = st.text_input("Comments (optional)", key="feedback_comment")
            with col2:
                rating = st.selectbox("Rating", [1, 2, 3, 4, 5], index=4)
            if st.button("Submit Feedback"):
                save_feedback(st.session_state['last_conv_id'], rating, comment)
                st.success("Thank you for your feedback!")
    
    with tab2:
        st.subheader("📊 Analytics Dashboard")
        
        analytics = get_analytics()
        conv_stats = analytics['conv_stats']
        
        if conv_stats.iloc[0]['total_conversations'] == 0:
            st.info("No conversations yet. Start chatting to see analytics!")
        else:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Conversations", int(conv_stats.iloc[0]['total_conversations']))
            col2.metric("Unique Users", int(conv_stats.iloc[0]['unique_users']))
            col3.metric("Avg Response Time", f"{conv_stats.iloc[0]['avg_response_time']:.2f}s")
            
            fallback_pct = analytics['fallback_rate'].iloc[0]['fallback_percentage']
            col4.metric("Fallback Rate", f"{fallback_pct:.1f}%", 
                       delta_color="inverse" if fallback_pct > 20 else "normal")
            
            # Charts row
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                st.subheader("Intent Distribution")
                intent_data = analytics['intent_dist']
                if len(intent_data) > 0:
                    fig = px.bar(intent_data, x='intent', y='count', 
                                color='count', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            
            with row1_col2:
                st.subheader("Conversations Over Time")
                timeline = analytics['timeline']
                if len(timeline) > 0:
                    fig = px.line(timeline, x='date', y='conversations', markers=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Bottom row
            row2_col1, row2_col2 = st.columns(2)
            
            with row2_col1:
                st.subheader("Satisfaction")
                sat = analytics['satisfaction']
                if sat.iloc[0]['total_feedback'] > 0:
                    st.metric("Avg Rating", f"{sat.iloc[0]['avg_rating']:.2f}/5")
                    st.progress(sat.iloc[0]['avg_rating'] / 5)
                    st.caption(f"Based on {int(sat.iloc[0]['total_feedback'])} ratings")
                else:
                    st.info("No feedback yet")
            
            with row2_col2:
                st.subheader("Top Unanswered Questions")
                unanswered = analytics['unanswered']
                if len(unanswered) > 0:
                    for _, row in unanswered.head(5).iterrows():
                        st.write(f"• {row['question']} ({row['count']} times)")
                else:
                    st.info("No unanswered questions tracked")
    
    with tab3:
        st.subheader("🧪 Intent Classification Evaluation")
        st.write("""
        Evaluate how well the intent classifier performs on a held-out test set.
        Add labeled examples to `intent_examples` table to build the test set.
        """)
        
        if st.button("Run Evaluation"):
            results = evaluate_intent_classification()
            
            if "error" in results:
                st.error(results["error"])
            else:
                st.success(f"Evaluation complete on {results['test_size']} test examples")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']:.2%}")
                col2.metric("Precision", f"{results['precision']:.2%}")
                col3.metric("Recall", f"{results['recall']:.2%}")
                col4.metric("F1 Score", f"{results['f1_score']:.2%}")
                
                st.subheader("Confusion Matrix")
                cm_df = pd.DataFrame(
                    results['confusion_matrix'],
                    index=results['labels'],
                    columns=results['labels']
                )
                fig = px.imshow(cm_df, text_auto=True, aspect='auto',
                               color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Recommendations")
                if results['accuracy'] < 0.8:
                    st.warning("Consider adding more training examples for underrepresented intents.")
                if results['recall'] < 0.7:
                    st.info("Some intents may need more distinctive examples.")

if __name__ == "__main__":
    main()