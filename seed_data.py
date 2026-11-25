"""
Seed data for intent classification training and evaluation.
Run this script to populate the database with initial training/test examples.
"""
import sqlite3
import os

DB_PATH = "chatbot_analytics.db"

# Training examples (80%)
TRAINING_EXAMPLES = [
    # greeting
    ("Hello there!", "greeting", False),
    ("Hi, how are you?", "greeting", False),
    ("Hey there!", "greeting", False),
    ("Good morning!", "greeting", False),
    ("Hello bot", "greeting", False),
    ("What's up?", "greeting", False),
    
    # product_info
    ("Tell me about your product", "product_info", False),
    ("What features do you offer?", "product_info", False),
    ("How does this work?", "product_info", False),
    ("What is your product about?", "product_info", False),
    ("Can you explain your features?", "product_info", False),
    ("What can this bot do?", "product_info", False),
    
    # pricing
    ("How much does it cost?", "pricing", False),
    ("What's the price?", "pricing", False),
    ("Tell me about pricing", "pricing", False),
    ("Are there any fees?", "pricing", False),
    ("How much is the subscription?", "pricing", False),
    ("What's the cost per month?", "pricing", False),
    
    # support
    ("I need help with something", "support", False),
    ("There's a problem", "support", False),
    ("It's not working", "support", False),
    ("I have an issue", "support", False),
    ("Can you assist me?", "support", False),
    ("I'm having trouble", "support", False),
    
    # faq
    ("Do you have a FAQ?", "faq", False),
    ("What are common questions?", "faq", False),
    ("Tell me about frequently asked questions", "faq", False),
    ("I have a question", "faq", False),
    
    # feedback
    ("I have feedback", "feedback", False),
    ("I'd like to give a suggestion", "feedback", False),
    ("You should improve this", "feedback", False),
    ("Here's my opinion", "feedback", False),
    
    # goodbye
    ("Goodbye", "goodbye", False),
    ("Thanks, bye", "goodbye", False),
    ("See you later", "goodbye", False),
    ("Thank you", "goodbye", False),
    ("That's all, thanks", "goodbye", False),
]

# Test examples (20%) - held out for evaluation
TEST_EXAMPLES = [
    ("Hello!", "greeting", True),
    ("What are the main features?", "product_info", True),
    ("How much money?", "pricing", True),
    ("Help me please", "support", True),
    ("Where can I find FAQs?", "faq", True),
    ("I want to leave feedback", "feedback", True),
    ("Bye for now", "goodbye", True),
    ("Hi there!", "greeting", True),
    ("What does this product do?", "product_info", True),
    ("Is it free or paid?", "pricing", True),
]

def seed_database():
    """Populate the database with training and test examples."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing examples
    cursor.execute("DELETE FROM intent_examples")
    
    # Insert training examples
    cursor.executemany(
        "INSERT INTO intent_examples (text, intent, is_test) VALUES (?, ?, ?)",
        TRAINING_EXAMPLES
    )
    
    # Insert test examples
    cursor.executemany(
        "INSERT INTO intent_examples (text, intent, is_test) VALUES (?, ?, ?)",
        TEST_EXAMPLES
    )
    
    conn.commit()
    conn.close()
    
    print(f"Added {len(TRAINING_EXAMPLES)} training examples and {len(TEST_EXAMPLES)} test examples.")

if __name__ == "__main__":
    seed_database()
    print("Database seeded successfully!")