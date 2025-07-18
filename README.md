### Documentation for Venice AI APIs

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

### 4. Using Venice AI's Utilities and Arguments

Utilities allow you to add additional features to your inference requests, such as database queries or external script execution. You can use the `chat.completions.create` method to send requests containing utilities.

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

### 5. Generating Images

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

### 6. Gradio space (for HuggingFace or not)

In the `/VeniceGradio.py` file, you'll find simple code to deploy an application that uses all the text models available on Venice.ai.

You'll just need an API key, as with everything else. 
Please note: the code is intended for personal use only (no concurrent conversations) and no permanent storage.

I personally use this to talk to an AI about topics without needing to keep the conversation.
