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

# Function to handle stream conversation
def run_conversation(prompt):
    """
    Runs a full conversation with streaming, tool call handling, and complete logging.
    Uses qwen3-235b for streaming reasoning and function calling.
    Handles edge cases in streaming chunks (empty choices, None content, etc.).
    Accumulates tool call arguments as list of chunks, joins at end for parsing.
    """
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Print system and user messages
    print(f"System: {system_prompt}")
    print(f"User: {prompt}")
    print("\nAssistant (streaming): ", end='', flush=True)
    
    # Streaming response with tool support
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.3,
        max_tokens=2000,
        stream=True,
        extra_body={"venice_parameters": {"include_venice_system_prompt": False}}
    )
    
    # Variables for aggregation
    full_response = ""
    accumulated_tool_calls = {}
    tool_calls = []
    in_tool_call = False
    arg_chunks = {}  # Per tool_call index: list of argument chunks

    # Process each chunk
    for chunk in stream:
        # Skip if no choices
        if not chunk.choices:
            continue
            
        delta = chunk.choices[0].delta
        if delta is None:
            continue

        # Handle content streaming
        if delta.content:
            if not full_response:  # First token
                print(delta.content, end='', flush=True)
                full_response += delta.content
            else:
                print(delta.content, end='', flush=True)
                full_response += delta.content

        # Handle tool calls in delta
        if delta.tool_calls:
            in_tool_call = True
            for tool_call in delta.tool_calls:
                index = tool_call.index
                if index not in accumulated_tool_calls:
                    accumulated_tool_calls[index] = {
                        "id": tool_call.id,
                        "function": {"name": "", "arguments": ""},
                        "type": tool_call.type,
                        "arg_chunks": []  # New: list for arguments
                    }
                if tool_call.function:
                    if tool_call.function.name:
                        accumulated_tool_calls[index]["function"]["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        # Append to list instead of +=
                        accumulated_tool_calls[index]["arg_chunks"].append(tool_call.function.arguments)

        # On finish reason: finalize tool calls
        if chunk.choices[0].finish_reason == "tool_calls":
            # Convert accumulated tool calls
            for idx, call in accumulated_tool_calls.items():
                # Join arg chunks
                joined_args = ''.join(call["arg_chunks"])
                # Clean up common malformations (e.g., leading/trailing {} or duplicates)
                # Simple fix: find the last complete JSON object
                if joined_args:
                    # Try to parse the full joined string
                    try:
                        parsed_args = json.loads(joined_args)
                        call["function"]["arguments"] = json.dumps(parsed_args)
                    except json.JSONDecodeError:
                        # Fallback: attempt to extract valid JSON substring
                        # Look for balanced braces starting from end
                        start = joined_args.rfind('{')
                        if start != -1:
                            potential_json = joined_args[start:]
                            try:
                                parsed_args = json.loads(potential_json)
                                call["function"]["arguments"] = json.dumps(parsed_args)
                            except json.JSONDecodeError:
                                # Last resort: use raw joined as string
                                call["function"]["arguments"] = joined_args
                        else:
                            call["function"]["arguments"] = joined_args
            
            tool_calls = [
                {
                    "id": call["id"],
                    "function": call["function"],
                    "type": call["type"]
                }
                for call in accumulated_tool_calls.values()
            ]
            
            # Display detected tool call
            for tc in tool_calls:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    print(f"\n\n[FUNCTION CALL: {tc['function']['name']} with {args}]")
                except json.JSONDecodeError:
                    print(f"\n\n[FUNCTION CALL: {tc['function']['name']} with invalid JSON: {tc['function']['arguments']}]")

            # Add assistant's first message (with tool calls) to messages
            messages.append({
                "role": "assistant",
                "content": full_response,
                "tool_calls": tool_calls
            })

            # Execute tool and get result
            for tc in tool_calls:
                if tc["function"]["name"] == "get_user_info":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                        result = get_user_info(args["name"])
                        # Append tool response to messages (after assistant message)
                        messages.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tc["id"],
                            "name": tc["function"]["name"]
                        })
                        print(f"[TOOL RESPONSE: {result}]")
                    except Exception as e:
                        print(f"[TOOL ERROR: {e}]")
                        # Append error as tool response to avoid breaking the chain
                        messages.append({
                            "role": "tool",
                            "content": f"Error: {e}",
                            "tool_call_id": tc["id"],
                            "name": tc["function"]["name"]
                        })

            # Now generate final response after tool execution
            print("\n\nAssistant (final response): ", end='', flush=True)
            final_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.6,
                max_tokens=2000,
                extra_body={"venice_parameters": {"include_venice_system_prompt": False}}
            )
            final_response = ""
            for final_chunk in final_stream:
                if not final_chunk.choices:
                    continue
                delta_final = final_chunk.choices[0].delta
                if delta_final is None:
                    continue
                if delta_final.content:
                    print(delta_final.content, end='', flush=True)
                    final_response += delta_final.content
            # Append final assistant message
            messages.append({"role": "assistant", "content": final_response})

    # If no tool call was made, just add the assistant response
    if not tool_calls:
        messages.append({"role": "assistant", "content": full_response})

    # Print full conversation at the end
    print("\n\n--- FULL CONVERSATION ---")
    for msg in messages:
        print(msg)

    return full_response if not tool_calls else final_response

# Test
run_conversation("What is Jean Dupont's position?")
