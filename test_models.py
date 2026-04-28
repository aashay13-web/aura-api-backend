import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your secret API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) # type: ignore

print("--- SCANNING YOUR UNLOCKED MODELS ---")
try:
    for m in genai.list_models(): # type: ignore
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error connecting to Google: {e}")