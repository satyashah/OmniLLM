import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
async def main():
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say this is a test"}]
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
