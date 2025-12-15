import os
import cohere
from dotenv import load_dotenv

load_dotenv()

api_key = "pKjuDdbT2kyfJsoZNkdRVOz63j2DQfBqOSbP9U2E"
if not api_key:
    print("No API key found")
else:
    co = cohere.Client(api_key)
    try:
        response = co.chat(
            message="Hello",
            model="command-a-03-2025"
        )
        print("command-a-03-2025 works!")
        print(response.text)
    except Exception as e:
        print(f"command-a-03-2025 failed: {e}")

    try:
        response = co.chat(
            message="Hello",
            model="command"
        )
        print("command works!")
        print(response.text)
    except Exception as e:
        print(f"command failed: {e}")
