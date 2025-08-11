#!/usr/bin/env python3
"""
Test script for local LLaMA LLM integration with Crew AI
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from my_pro.tools.local_llm import local_llm

def test_local_llm():
    """Test the local LLM integration."""
    print("🧪 Testing Local LLaMA LLM Integration")
    print("=" * 50)
    
    # Check if model is loaded
    print(f"Model loaded: {local_llm.is_model_loaded()}")
    
    if local_llm.is_model_loaded():
        # Get model info
        info = local_llm.get_model_info()
        print(f"Model info: {info}")
        
        # Test completion
        print("\n📝 Testing text completion...")
        prompt = "What is artificial intelligence?"
        response = local_llm.complete(prompt, max_tokens=100)
        print(f"Prompt: {prompt}")
        print(f"Response: {response.content}")
        
        # Test chat
        print("\n💬 Testing chat completion...")
        messages = [{"role": "user", "content": "Explain machine learning in simple terms"}]
        chat_response = local_llm.chat(messages, max_tokens=150)
        print(f"Chat response: {chat_response.content}")
        
    else:
        print("❌ Model failed to load. Please check:")
        print("1. Model path exists")
        print("2. Required dependencies are installed")
        print("3. Model files are valid")

if __name__ == "__main__":
    test_local_llm()
