from openai import OpenAI
import json

# Set up OpenAI client
client = OpenAI(
    api_key="api_key",
    base_url="https://api.venice.ai/api/v1"
)

# User database
user_database = {
    "jean dupont": {
        "age": 45,
        "position": "Director Marketing",
        "company": "TechCorp",
        "city": "Paris"
    }
}

# System prompt
system_prompt = """You are a technical assistant who MUST use the get_user_info tool before answering questions about users. The database only contains Jean Dupont."""

# Tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_user_info",
        "description": "Access the employee database",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
    }
}]

# Function to get user info
def get_user_info(name):
    return user_database.get(name.strip().lower(), "Not found")

# Function to handle stream
def handle_stream(stream):
    full_response = ""
    print("Assistant: ", end='', flush=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_response += content
    return full_response

# Function to run conversation
def run_conversation(prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # First step: Tool detection without streaming
    response = client.chat.completions.create(
        model="llama-3.3-70b",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
        max_tokens=500,
        extra_body={"venice_parameters": {"include_venice_system_prompt": False}}
    )
    
    msg = response.choices[0].message
    tool_calls = msg.tool_calls
    
    if tool_calls:
        # Execute tool
        for call in tool_calls:
            if call.function.name == "get_user_info":
                args = json.loads(call.function.arguments)
                result = get_user_info(args["name"])
                
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": call.id,
                    "name": "get_user_info"
                })
        
        # Second step: Final response in streaming
        final_stream = client.chat.completions.create(
            model="llama-3.2-3b",
            messages=messages,
            stream=True,
            temperature=0.5,
            max_tokens=500
        )
        return handle_stream(final_stream)
    
    else:
        # Direct response in streaming if no tool
        direct_stream = client.chat.completions.create(
            model="llama-3.2-3b",
            messages=messages,
            stream=True,
            temperature=0.5,
            max_tokens=500
        )
        return handle_stream(direct_stream)

# Test
print(run_conversation("What is Jean Dupont's position?"))
print("\n---\n")
print(run_conversation("Who is Marie Curie?"))