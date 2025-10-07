### Extra

* Go to [open-taranis](https://github.com/SyntaxError4Life/open-taranis) to see more and future uses of the Venice.ai API

### Documentation for Venice AI APIs

- [1. Getting the List of Available Models](#1-getting-the-list-of-available-models)
- [2. Inferring via API in Normal Mode](#2-inferring-via-api-in-normal-mode)
- [3. Inferring via API in Streaming Mode](#3-inferring-via-api-in-streaming-mode)
- [4. Using Venice AI's tools call](#4-using-venice-ais-tools-call)
- [5. Generating images](#5-generating-images)
- [6. Gradio space](#6-gradio-space-for-huggingface-or-not)
- [7. Configuration in brave leo](#7-configuration-in-brave-leo)

Venice.Ai API Specifics :
- `tool_calls` in streaming: arrives incrementally — requires manual aggregation of `index`, `id`, `name`, and `arguments`
- `finish_reason="tool_calls"` comes in a **separate chunk** after the deltas (not in the same chunk) and only for less than two tool call
- Streaming chunks may be **empty or metadata-only** — always check `choices` and `delta` before access
- `extra_body` required for model control (e.g. `"venice_parameters": {"include_venice_system_prompt": False}`)

Additional files :
- [Standard chat loop w/ streaming tokens](https://github.com/SyntaxError4Life/Venice.ai-API/blob/main/VeniceChat.py)
- [Full Streaming Call Tools](https://github.com/SyntaxError4Life/Venice.ai-API/blob/main/VeniceTools.py)

---

### 1. Getting the List of Available Models

To get the list of available models via the Venice AI API, you can use the `models.list` method of the OpenAI library.

```python
from openai import OpenAI

# Configuration of the client with the API key and the Venice base URL
client = OpenAI(
    api_key="api_key",
    base_url="https://api.venice.ai/api/v1"
)

# Get the list of available models
models = client.models.list()

# Print the names of the models
for model in models.data:
    print(model.id)
```

Expected output:

```
venice-uncensored
qwen-2.5-qwq-32b
qwen3-4b
mistral-31-24b
qwen3-235b
llama-3.2-3b
llama-3.3-70b
llama-3.1-405b
dolphin-2.9.2-qwen2-72b
qwen-2.5-vl
qwen-2.5-coder-32b
deepseek-r1-671b
deepseek-coder-v2-lite
```

---

### 2. Inferring via API in Normal Mode

To perform inference in normal mode, you can use the `chat.completions.create` method to generate text based on a given prompt. You can also specify additional parameters such as `temperature`, `max_tokens`, etc., as well as Venice AI-specific parameters via `extra_body`.

```python
from openai import OpenAI

client = OpenAI(
    api_key="api_key",
    base_url="https://api.venice.ai/api/v1"
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    frequency_penalty=0.2,
    presence_penalty=0.2,
    max_tokens=50,
    extra_body={
        "venice_parameters": {
            "include_venice_system_prompt": False
        }
    }
)

print(response.choices[0].message.content)
```

---

### 3. Inferring via API in Streaming Mode

To perform inference in streaming mode, you can activate the `stream=True` parameter. This allows you to receive responses progressively, which is useful for real-time applications.

```python
from openai import OpenAI
client = OpenAI(
    api_key="api_key",
    base_url="https://api.venice.ai/api/v1"
)

stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.7,
    frequency_penalty=0.2,
    presence_penalty=0.2,
    max_tokens=50,
    stream=True,  # Activate streaming
    extra_body={
        "venice_parameters": {
            "include_venice_system_prompt": False
        }
    }
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        text = chunk.choices[0].delta.content
        print(text, end='', flush=True)
```

---

### 4. Using Venice AI's tools call

Tools call allow you to add additional features to your inference requests, such as database queries or external script execution. You can use the `chat.completions.create` method to send requests containing utilities.

```python
from openai import OpenAI

client = OpenAI(
    api_key="api_key",
    base_url="https://api.venice.ai/api/v1"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current time",
            "parameters": {
                "type": "object",  # Required schema even if empty
                "properties": {}
            }
        }
    }
]

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "Your system prompt"},
        {"role": "user", "content": "What time is it now?"}
    ],
    tools=tools,
    tool_choice="auto",  # Optional but recommended for control
    extra_body={
        "venice_parameters": {
            "include_venice_system_prompt": False
        }
    }
)

# Check for function calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function called: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
else:
    print(response.choices[0].message.content)
```

---

### 5. Generating images

To generate images via the Venice AI API, you can use the `/image/generate` endpoint directly. Here's an example of code to generate an image and save it locally.

```python
from openai import OpenAI
import base64

client = OpenAI(
    api_key="api_key",
    base_url="https://api.venice.ai/api/v1"
)

# Image generation parameters
params = {
    "model": "flux-dev",
    "prompt": "An image of a cat sitting on a wall.",
    "width": 1024,
    "height": 1024,
    "steps": 6,
    "hide_watermark": True,
    "return_binary": False,  # To get the image in base64
    "seed": 123,
    "cfg_scale": 7.5,
    "style_preset": "3D Model",
    "negative_prompt": "",
    "safe_mode": False
}

# Directly call the Venice endpoint
response = client._client.post(
    "/image/generate",
    json=params,
    headers={"Authorization": f"Bearer {client.api_key}"}
)

# Decode and save the image
image_data = base64.b64decode(response.json()["images"][0])
with open("generated_image.png", "wb") as f:
    f.write(image_data)

print("Image generated successfully!")
```

---

### 6. Gradio interface

In the [/VeniceGradio.py](https://github.com/SyntaxError4Life/Venice.ai-API/blob/main/VeniceGradio.py) file, you'll find simple code to deploy an application that uses all the text models available on Venice.ai.

You'll just need an API key, as with everything else.

Please note: the code is intended for personal use only (no concurrent conversations) and no permanent storage.

I personally use this to talk to an AI about topics without needing to keep the conversation.

---

### 7. Configuration in [brave leo](brave://settings/leo-ai)

Exemple for a model :
- Name : `Qwen3 235b`
- Request name : `qwen3-235b`
- Endpoint : `https://api.venice.ai/api/v1/chat/completions`
- Context : `32000`
- APK_KEY : [here](https://venice.ai/settings/api)
- System prompt : **what u want**
- Image : `no`


