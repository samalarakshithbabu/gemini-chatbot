"""Test script for NEXORA AI chatbot"""
from nexora_ai import GoogleGenAIClient
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY', '')
client = GoogleGenAIClient(api_key)

print("=" * 60)
print("NEXORA AI - MANDATORY CHATBOT TESTS")
print("=" * 60)
print()

# Test 1: Hi
print("TEST 1: Enter 'Hi'")
print("-" * 40)
response, success = client.generate_content("Hi")
passed = "hello" in response.lower() or "hi" in response.lower()
print(f"Expected: AI responds with greeting")
print(f"Actual: {response}")
print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'}")
print()

# Test 2: What is Python?
print("TEST 2: Enter 'What is Python?'")
print("-" * 40)
response, success = client.generate_content("What is Python?")
passed = "python" in response.lower() and "programming" in response.lower()
print(f"Expected: Genuine AI explanation about Python")
print(f"Actual: {response[:150]}...")
print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'}")
print()

# Test 3: Explain SQL joins
print("TEST 3: Enter 'Explain SQL joins'")
print("-" * 40)
response, success = client.generate_content("Explain SQL joins")
passed = "join" in response.lower() and "sql" in response.lower()
print(f"Expected: Technical explanation of SQL JOINs")
print(f"Actual: {response[:150]}...")
print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'}")
print()

# Test 4: Artificial Intelligence
print("TEST 4: Enter 'What is Artificial Intelligence?'")
print("-" * 40)
response, success = client.generate_content("What is Artificial Intelligence?")
passed = "intelligence" in response.lower() or "ai" in response.lower()
print(f"Expected: Detailed answer about AI")
print(f"Actual: {response[:150]}...")
print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'}")
print()

# Test 5: Follow-up question (context)
print("TEST 5: Follow-up question")
print("-" * 40)
context = """Previous conversation:
User: What is Python?
Bot: Python is a popular high-level programming language...

User: How do I add items to it?"""
response, success = client.generate_content(context)
passed = "append" in response.lower() or "add" in response.lower() or "list" in response.lower()
print(f"Expected: AI understands context about Python lists")
print(f"Actual: {response[:150]}...")
print(f"RESULT: {'✓ PASS' if passed else '✗ FAIL'}")
print()

print("=" * 60)
print(f"MODEL USED: {client._get_working_model()}")
print("=" * 60)