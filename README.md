# nonchalant_chatbot
AI chatbot for fyp


# AI Chatbot Module – AMS Enhancement

This document explains how to install, configure, and run the AI-powered chatbot module for the Asset Management System (AMS). This includes both the fine-tuned NER model and the optional LLM integration.

---

## 1. Prerequisites

Before installation, ensure the following are available:

### Server Requirements
- OS: Ubuntu 20.04+ / Windows WSL / macOS
- Python 3.9+
- Laravel app (running on `localhost:8000` or configured host)
- Redis (for context memory)
- Optional: Docker (if using Ollama for LLM)

### Installed Tools
```bash
sudo apt install redis
pip install virtualenv
```

---

## 2. Project Setup

Clone or download the chatbot AI module into your Laravel project directory:

```bash
git clone <this-repo-url> ai-chatbot
cd ai-chatbot
```

Create a Python virtual environment:

```bash
python -m venv chat_env
source chat_env/bin/activate  # Windows: chat_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Load the Fine-Tuned NER Model

Ensure your custom fine-tuned model is saved in a folder like:

```
./fine_tuned_ner/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.txt
...
```

In `main.py`, set the model path:

```python
model_path = "./fine_tuned_ner"
```

---

## 4. (Optional) Install Ollama for LLM Support

> Only needed if you want to enable `mode=llm` queries.

```bash
curl https://ollama.com/install.sh | sh
ollama run mistral
```

Ensure Mistral is running locally (`http://localhost:11434` by default).

---

## 5. Run the AI API

Start the FastAPI service (runs on port 8001):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

You should see:

```
NER pipeline loaded from ./fine_tuned_ner
Using Ollama's Mistral model via API
Uvicorn running on http://0.0.0.0:8001
```

---

## 6. Connect Laravel to AI API

Laravel connects via `POST /query` to:

```
http://localhost:8001/query
```

Required payload:

```json
{
  "question": "Show me assets in HQ",
  "sid": "session_id_string",
  "mode": "ner"  // or "llm"
}
```

---

## 7. Modes of Operation

| Mode   | Behavior                          |
|--------|-----------------------------------|
| `ner`  | Uses BERT NER model + rule logic  |
| `llm`  | Uses Mistral (Ollama) for fallback|
| `rule` | Rule-only (fallback/debug)        |

---

## 8. Test Examples

```bash
curl -X POST localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me assets in HQ", "sid": "abc123", "mode": "ner"}'
```

---

## 9. Logs for Debugging

Console output includes:
- NER token-level predictions
- Rule fallback triggers
- Intent routing
- Final filters passed to Laravel

---

## 10. Common Issues

| Problem                       | Fix |
|------------------------------|-----|
| `Model path not found`       | Check `model_path` in `main.py` |
| `Ollama connection error`    | Ensure Mistral is running locally |
| `Redis not running`          | Start Redis with `sudo service redis start` |

---

## 11. Deploying on an AMS Server (Production Mode)

To deploy the AI chatbot as a service alongside your Laravel-based AMS on a cloud or internal server (e.g. AWS EC2, DigitalOcean, or local VM), follow the steps below:

### Recommended Directory Structure

```
/var/www/ams/
├── laravel-app/
├── ai-chatbot/
│   ├── main.py
│   ├── app/
│   ├── fine_tuned_ner/
│   └── requirements.txt
```

### Environment Configuration

Create a `.env` file or set environment variables to control sensitive or deployment-specific values:

```bash
export MODEL_PATH=/var/www/ams/ai-chatbot/fine_tuned_ner
export PORT=8001
export REDIS_URL=redis://localhost:6379
```

### Run as a Background Service (Systemd)

Create a service file at `/etc/systemd/system/ai-chatbot.service`:

```ini
[Unit]
Description=AI Chatbot API for AMS
After=network.target

[Service]
User=www-data
WorkingDirectory=/var/www/ams/ai-chatbot
ExecStart=/var/www/ams/ai-chatbot/chat_env/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reexec
sudo systemctl enable ai-chatbot
sudo systemctl start ai-chatbot
```

### Exposing via NGINX (Recommended)

Set up reverse proxy if needed for CORS control or HTTPS:

```nginx
server {
    listen 80;
    server_name ai.ams.example.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Laravel Configuration (Remote URL)

In Laravel `.env`, update the chatbot service endpoint:

```
CHATBOT_API_URL=http://ai.ams.example.com/query
```

Make sure CORS and CSRF settings are aligned in Laravel and FastAPI.

### Health Check Route (Optional)

You can expose a simple endpoint for monitoring:

```python
@app.get("/health")
def health_check():
    return {"status": "OK"}
```


## Maintainer

**Min Thaw Khant**  
Group 9 — AMS Enhancement Team  
Email: *[andymoriarty12@gmail.com]*
