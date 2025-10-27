hall_evaluation_prompt = """
Instruction: You will be given part of a dialogue between the user and the system. 
In this dialogue, the user is requesting information from the system, and the system will execute an API call to retrieve the necessary information.
Your task is to generate a response that satisfies the user's initial query based on the API call results provided in the dialogue history.

Dialogue History:

"""

add_tokens = ['<start_of_retriever_status>', '<end_of_retriever_status>','<end_of_user_message>',
'<end_of_system_action>','<end_of_dialogue_state>','<end_of_call_result>',
'<start_of_user_message>','<end_of_system_message>','<end_of_system_thought>',
'<start_of_dialogue_state>','<start_of_api_documentation>','<start_of_system_action>',
'<start_of_system_message>','<end_of_api_documentation>','<start_of_call_result>','<start_of_system_thought>']

evaluation_dst_prompt = """
Instruction: You will be given part of a dialogue between the user and the system. In this dialogue, the user is requesting information from the system, and the system will execute an API call to retrieve the necessary information.

Your task is to output the appropriate dialogue state for the current turn, based on the dialogue provided.

System Rules:
1. The system selects the API with the highest score from among the APIs in the retriever status that have a score of 0.6 or higher, which is suitable for processing the user's query.
2. If no APIs score is higher than 0.6, system cannot confirm the API to call.

Dialogue state format:

Case1. When the API has not been confirmed(If the retrived api does not have an API with a score of 0.6 or higher):
{'api_confirmed': 'false', 'api_status': 'none'}
- The API is not confirmed, so api_confirmed is set to false.
- Therefore, api_status is none.
- If the api_confirmed is false, api_status must be none.

Case2. When the API is confirmed(If the retrived API has an API with a score of 0.6 or higher):
{'api_confirmed': 'true', 'api_status': {'api_name': 'API1', 'required_parameters': {'param1': '', 'param2': 'value1'}, 'optional_parameters': {'param3': ''}}
- The API is confirmed, so api_confirmed is set to true.
- api_status contains the name of the API and the input parameter list needed for the API call. Any parameter whose value can be extracted from the dialogue history will have its value filled in.
- Case2's 'param1','param2','param3' is just an example value. Don't use this parameters. Refer the given API documentation on each turn.
- The input parameter should always be determined by looking at API documentation. Don't hallucinate it.
    
Now, part of the dialogue will be give. Just generate the dialogue state in given format. Not a single words.

Dialogue:

"""

action_evaluation_prompt_thought_normal = """
Instruction: You will be given part of a dialogue between the user and the system. 
In this dialogue, the user is requesting information from the system, and the system will execute an API call to retrieve the necessary information.
Your task is to predict the action that the system should take after the last utterence of the user.

Read Dialogue history and return one action that is most appropriate for the system to take next.
The actions that the system can take are as follows.
- Request: In a situation where the API to be executed has been determined, ask the user questions to gather information on the input parameters.
- Response: Reply to user's request based on the result of API call.
- Clarify: If user's query is vague, re-ask user to get intent specifically. If there is no API in the most recently retrieved results with a score above 0.5, "clarify" is required.
- Suggest: Making a suggestion for an unclear user's intent and ask-ing whether it satisfies the user. If there is an API in the most recently retrieved results with a score above 0.5 but none exceeding 0.6, a 'suggest' action is required.
- Response fail: Notify the user that the system cannot execute the request due to insufficient information.
- System bye: System says goodbye to user politely.
- Call: Call the API with collected information from user or else and don't reply to user yet.
- Retriever call: Call the retriever to find proper API to satisfy user's requests. The system should call the retriever in the following two situations:
1. When the user specifies a requirement, and the system needs to search for an API to fulfill it.
2. When the user does not provide input parameters required for an API call, and the system needs to search for another API to obtain those parameters.

First, generate an appropriate 'thought' within the current dialogue context and the system's internal reasoning steps, and then select the most suitable action from the eight actions provided.
Do not return any value other than the action given above.
The return format is as follows.

Return format:
- Thought: (System's thought for the next action)
- Action:(predicted action)

Just return the answer with given format. Not a single words.

Dialogue History:

"""


action_evaluation_prompt_thought_token = """
Instruction: You will be given part of a dialogue between the user and the system. 
In this dialogue, the user is requesting information from the system, and the system will execute an API call to retrieve the necessary information.
Your task is to predict the action that the system should take after the last utterence of the user.

Read Dialogue history and return one action that is most appropriate for the system to take next.
The actions that the system can take are as follows.
- Request: In a situation where the API to be executed has been determined, ask the user questions to gather information on the input parameters.
- Response: Reply to user's request based on the result of API call.
- Clarify: If user's query is vague, re-ask user to get intent specifically. If there is no API in the most recently retrieved results with a score above 0.5, "clarify" is required.
- Suggest: Making a suggestion for an unclear user's intent and ask-ing whether it satisfies the user. If there is an API in the most recently retrieved results with a score above 0.5 but none exceeding 0.6, a 'suggest' action is required.
- Response fail: Notify the user that the system cannot execute the request due to insufficient information.
- System bye: System says goodbye to user politely.
- Call: Call the API with collected information from user or else and don't reply to user yet.
- Retriever call: Call the retriever to find proper API to satisfy user's requests. The system should call the retriever in the following two situations:
1. When the user specifies a requirement, and the system needs to search for an API to fulfill it.
2. When the user does not provide input parameters required for an API call, and the system needs to search for another API to obtain those parameters.

First, generate an appropriate 'thought' within the current dialogue context and the system's internal reasoning steps, and then select the most suitable action from the eight actions provided.
Do not return any value other than the action given above.
The return format is as follows.

Return format:
<start_of_system_thought> (System's thought for the next action) <end_of_system_thought>
<start_of_system_action> (predicted action) <end_of_system_action><|eot_id|>

Just return the answer with given format. Not a single words.

Dialogue History:

"""

action_evaluation_prompt_action = """
Instruction: You will be given part of a dialogue between the user and the system. 
In this dialogue, the user is requesting information from the system, and the system will execute an API call to retrieve the necessary information.
Your task is to predict the action that the system should take after the last utterence of the user.

Read Dialogue history and return one action that is most appropriate for the system to take next.
The actions that the system can take are as follows.
- Request: Asks some information to user.
- Response: Reply to user's request based on the result of API call.
- Clarify: If user's query is vague, re-ask user to get intent specifically. If there is no API in the most recently retrieved results with a score above 0.5, "clarify" is required.
- Suggest: Making a suggestion for an unclear user's intent and ask-ing whether it satisfies the user. If there is an API in the most recently retrieved results with a score above 0.5 but none exceeding 0.6, a 'suggest' action is required.
- Response fail: Notify the user that the system cannot execute the request due to insufficient information.
- System bye: System says goodbye to user politely.
- Call: Call the API with collected information from user or else and don't reply to user yet.
- Retriever call: Call the retriever to find proper API to satisfy user's requests. The system should call the retriever in the following two situations:
1. When the user specifies a requirement, and the system needs to search for an API to fulfill it.
2. When the user does not provide input parameters required for an API call, and the system needs to search for another API to obtain those parameters.

Of the eight actions given, return only the action that you think is most appropriate. Do not return any value other than the action given above.
Just return the action, not a single words.

Dialogue History:

"""

ECG_EVALUATION_PROMPT = """
Instruction:
You will be provided with part of a dialogue between the user and the system. The dialogue is conducted with a ECG wearable device user and the system, therefore the system should provide direct answers to the user's inquries without acting like a medical professional.
In this dialogue, the user is inquiring about an ECG reading, and the system will either respond directly or make appropriate tool calls to retrieve the necessary information to respond.
Your task is to generate appropriate thoughts, actions, and responses based on the dialogue history and the user's most recent utterance.

<Action list>
- call_classification_tool: Use this to identify arrhythmias, abnormalities, and findings from the provided ECG data
- call_measurement_tool: Use this to measure and output the heart rate, PR interval, QRS duration, and QTc interval.
- response: Provide comprehensive answer using results from tool outputs, combining technical findings with clinical interpretation
- response_fail: Indicate that the requested analysis cannot be performed due to tool limitations or because the question is not within the scope of the System and requires medical professional consultation.
- response_followup: Responds to the user's followup questions by providing additional information, clarification, or related insights about previous discussed ECG findings without requiring new tool calls.
- system_bye: Acknowledges the user's gratitude end the conversation politely

<General Rules>
1. Each generated dialogue (except for system_bye) is during the conversation with the user, therefore the responses should only address the user's last utterance and should not be too long.
2. Generated dialogue should be like a natural conversation between a user and ECG wearable assistant.

Based on the rules above, you will get an input as below:
Input:
<Dialogue history>
User: <User's last utterance> or Assistant: <Assistant's last Tool_Output>

And here is the format for what you should return in two different cases:

Case1. When a tool must be called based on the user's inquiry:
Action: <Assistant's chosen action, one of the tools>
Thought: <Assistant's reasoning process>
Tool_Output: <This will be provided externally.>

Case2. When providing a response based on previous tool or any other action not requiring a tool call:
Action: <Assistant's chosen action>
Thought: <Assistant's reasoning process>
Content: <Assistant's response>

Now, generate appropriate reasoning trace and responding message to user. Only generate in the proper format above, without indicating the chosen case.
"""

overall_evaluation_prompt = """
Instruction:
You will be provided with part of a dialogue between the user and the system. In this dialogue, the user is requesting information from the system, and the system will make an API call to retrieve the necessary information.
Your task is to generate an appropriate dialogue state, reasoning, and action based on the dialogue history and the user's most recent utterance. Additionally, generate an appropriate response message for the user's last utterance.

---------------Refer the internal rule----------------------

<Retriever status rules>
1. The system selects the API with the highest score from among the APIs in the retriever status that have a score of 0.6 or higher, which is suitable for processing the user's query.
2. If no APIs score is higher than 0.6, system cannot confirm the API to call.

<Dialogue state format>
Case1. When the API has not been confirmed(If the retrived api does not have an API with a score of 0.6 or higher):
{'api_confirmed': 'false', 'api_status': 'none'}
- The API is not confirmed, so api_confirmed is set to false.
- Therefore, api_status is none.
- If the api_confirmed is false, api_status must be none.

Case2. When the API is confirmed(If the retrived API has an API with a score of 0.6 or higher):
{'api_confirmed': 'true', 'api_status': {'api_name': 'API1', 'required_parameters': {'param1': '', 'param2': 'value1'}, 'optional_parameters': {'param3': ''}}
- The API is confirmed, so api_confirmed is set to true.
- api_status contains the name of the API and the input parameter list needed for the API call. Any parameter whose value can be extracted from the dialogue history will have its value filled in.
- Case2's 'param1','param2','param3' is just an example value. Don't use this parameters. Refer the given API documentation on each turn.
- The input parameter should always be determined by looking at API documentation. Don't hallucinate it.

<Action list>
The actions that the system can take are as follows.
- Request: In a situation where the API to be executed has been determined, ask the user questions to gather information on the input parameters.
- Response: Reply to user's request based on the result of API call.
- Clarify: If user's query is vague, re-ask user to get intent specifically. If there is no API in the most recently retrieved results with a score above 0.5, "clarify" is required.
- Suggest: Making a suggestion for an unclear user's intent and ask-ing whether it satisfies the user. If there is an API in the most recently retrieved results with a score above 0.5 but none exceeding 0.6, a 'suggest' action is required.
- Response fail: Notify the user that the system cannot execute the request due to insufficient information.
- System bye: System says goodbye to user politely.
- Call: Call the API with collected information from user or else and don't reply to user yet.
- Retriever call: Call the retriever to find proper API to satisfy user's requests. The system should call the retriever in the following two situations:
1. When the user specifies a requirement, and the system needs to search for an API to fulfill it.
2. When the user does not provide input parameters required for an API call, and the system needs to search for another API to obtain those parameters.

----------------------------------

Based on the rules above, you will get an input as below:
Input:
<Dialogue history>
User: <User's last utterance>

And here is the format for what you should return:
- Dialogue state: <First dialogue state after the user's last utterance. You generate here>
- Thought: <System's thought process. You generate here>
- Action: <System's action. You generate here. Choose one from <Action list>>
- Observation: <This will be provided externally.>
- Thought: <System's thought process. You generate here, refering the dialogue history and the observation above>
- Action: <System's action. You generate here, refering the dialogue history and the observation above. Choose one from <Action list>>
- Observation: <This will be provided externally.>
...
- Dialogue state: <Second dialogue state after the user's last utterance. You generate here>
- Message: <System's message>

Output Format Rules:
1. Always include two dialogue states in the output:
- The first one immediately after the user's last utterance.
- The second one just before the system's message.
2. Multiple reasoning steps (Thought-Action pairs) may be present in the output if required

Now, generate appropriate reasoning trace and responding message to user.

Dialogue History:
"""

geval_prompt = """

You will be given a conversation between two individuals. 

Read the User and System conversation and evaluate whether or not the response of the last system was answered based on the API call result.

Your task is to rate the responses on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Crieteria:

Correctness (True or False) Is the System answering the user based on the result of API call?

- True (correct) means that all information or facts in the response of the system match the result of API call.
- False (incorrect) means that any information in the response of the system or any information that is inconsistent with the result of API call or that is not in the result of API call.

Evaluation Steps:

1. Read the User and System Conversation: Review the conversation thoroughly, including the user's request and the system's response. Pay attention to the flow and details of the dialogue to understand the context and user intent.
2. Check the API Call Result: Review the result of the API call that the system used to generate its response. Make sure you fully understand the details and data returned by the API.
3. Compare the System’s Response to the API Result: Analyze the system's response in the conversation and cross-check it with the API call result. Ensure that all the facts and information presented in the response are directly supported by the API call output.
4. Assign a Correctness Rating: Mark True if the system’s response correctly aligns with the API call result, presenting all relevant information accurately. Mark False if any part of the system’s response contains incorrect information, omits critical details from the API result, or includes details not supported by the API call.
5. Submit the Rating: Provide a rating of True or False only, without any additional commentary.

Dialogue History:
{{Response}}

Evaluation Form (return only "True" or "False"):

- Correctness:
"""


def post_process(text):
    if "```json" in text:
        text = text.replace("```json","")
    if "```" in text:
        text = text.replace("```", "")
    return text
