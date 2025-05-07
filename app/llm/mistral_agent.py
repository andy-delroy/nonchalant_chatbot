import httpx
#httpx is an async-capable client library (like reqeusts but supports async/await). Like axios equivalent in react

OLLAMA_API_URL = "http://localhost:11434/api/generate"
#this is the local default ollama endpoint

MODEL_NAME = "mistral"

async def query_mistral(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    # this is the json body that is sent to Ollama's /generate endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"]
        #parses the JSON returned by Ollama and returns the "response" filed which is Mistral's answer to ur prompt
