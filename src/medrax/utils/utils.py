import os
import json
from typing import Dict, List


def load_prompts_from_file(file_path: str) -> Dict[str, str]:
    """
    Load multiple prompts from a file.

    Args:
    file_path (str): Path to the file containing prompts.

    Returns:
    Dict[str, str]: A dictionary of prompt names and their content.

    Raises:
    FileNotFoundError: If the specified file is not found.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    prompts = {}
    current_prompt = None
    current_content = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                if current_prompt:
                    prompts[current_prompt] = "\n".join(current_content).strip()
                current_prompt = line[1:-1]
                current_content = []
            elif line:
                current_content.append(line)

    if current_prompt:
        prompts[current_prompt] = "\n".join(current_content).strip()

    return prompts


def load_tool_prompts(tools: List[str], tools_json_path: str) -> str:
    """
    Load prompts for specified tools from the tools.json file.

    Args:
    tools (List[str]): List of tool names to load prompts for.
    tools_json_path (str): Path to the tools.json file.

    Returns:
    str: A string containing prompts for the specified tools.

    Raises:
    FileNotFoundError: If the tools.json file is not found.
    """
    if not os.path.exists(tools_json_path):
        raise FileNotFoundError(f"Tools JSON file not found: {tools_json_path}")

    with open(tools_json_path, "r") as file:
        tools_data = json.load(file)

    tool_prompts = []
    for tool in tools:
        if tool in tools_data:
            tool_info = tools_data[tool]
            tool_prompt = f"Tool: {tool}\n"
            tool_prompt += f"Description: {tool_info['description']}\n"
            tool_prompt += f"Usage: {tool_info['prompt']}\n"
            tool_prompt += f"Input type: {tool_info['input_type']}\n"
            tool_prompt += f"Return type: {tool_info['return_type']}\n\n"
            tool_prompts.append(tool_prompt)

    return "\n".join(tool_prompts)


def load_system_prompt(
    system_prompts_file: str,
    system_prompt_type: str,
    tools: List[str],
    tools_json_path: str,
) -> str:
    """
    Load the system prompt by combining the system prompt and tool information.

    Args:
    system_prompts_file (str): Path to the file containing system prompts.
    system_prompt_type (str): The type of system prompt to use.
    tools (List[str]): List of tool names to include in the prompt.
    tools_json_path (str): Path to the tools.json file.

    Returns:
    str: The system prompt combining system prompt and tool information.
    """
    prompts = load_prompts_from_file(system_prompts_file)
    system_prompt = prompts.get(system_prompt_type, "GENERAL_ASSISTANT")
    tool_prompts = load_tool_prompts(tools, tools_json_path)

    return f"{system_prompt}\n\nTools:\n{tool_prompts}".strip()
