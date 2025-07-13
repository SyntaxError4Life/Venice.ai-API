import os
import gradio as gr
from openai import OpenAI

# Venice client
client = OpenAI(
    api_key="api_key", # or os.getenv("API_KEY") for HF/env
    base_url="https://api.venice.ai/api/v1"
)

# Load available models once at start-up
try:
    MODELS = sorted(m.id for m in client.models.list().data)
except Exception:
    MODELS = ["venice-uncensored"]   # fallback

# Gradio chat function
#   Gradio sends:  message, history, model
def chat_fn(message, history, model):
    """
    history : list of [user, assistant] pairs
    We rebuild the full chat-format history every turn.
    """
    # Convert history to OpenAI dicts
    messages = []
    for h_user, h_bot in history:
        messages.append({"role": "user", "content": h_user})
        messages.append({"role": "assistant", "content": h_bot})
    messages.append({"role": "user", "content": message})

    # Stream reply
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_body={"venice_parameters": {"include_venice_system_prompt": False}}
    )

    partial = ""
    for chunk in stream:
        token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
        partial += token
        yield partial

# Gradio UI
gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[
        gr.Dropdown(choices=MODELS, value=MODELS[0], label="Model")
    ],
    title="Venice Streaming Chat"
).launch()