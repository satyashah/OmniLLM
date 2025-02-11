from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv


"""
V1

The agent will receive a prompt from the user

The agent will determine the characteristics of the prompt (language, subject, math, factual, etc.)

A script balancing between the characteristics (and cost) will be used to select the best LLM for the prompt

The selected LLM will be called with the prompt

The response will be returned to the user
"""

"""
V2

The agent will receive a prompt from the user

The agent will determine the characteristics of the prompt (language, subject, math, factual, etc.)

A script balancing between the characteristics (and cost) will be used to select the best n LLMs for the prompt

The selected LLMs will be called by the agent

The responses will be aggregated and returned to the user
"""






load_dotenv()



def gpt(prompt: str) -> str:
    """
    Use GPT to answer a question.
    
    GPT Stats:
    - Model: gpt-4o
    - Math Skills: 80%
    - Language Skills: 90%
    - General Knowledge and Facts: 80%
    """

    return "GPT answered the question."


def claude(prompt: str) -> str:
    """
    Use Claude to answer a question.

    Claude Stats:
    - Model: Claude-4o
    - Math Skills: 90%
    - Language Skills: 80%
    - General Knowledge and Facts: 80%
    """


    return "Claude answered the question."

tools = [gpt, claude]

# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with determining which LLM to use to answer a question. Only call one tool then END.")


# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()

# Run graph
result = graph.invoke({"messages": [HumanMessage(content="Write me a blog about the latest news in technology")]})

# Print cleaner output
for message in result["messages"]:
    if isinstance(message, HumanMessage):
        print(f"Human: {message.content}")
    elif isinstance(message, AIMessage):
        if message.content:
            print(f"Assistant: {message.content}")
        if message.additional_kwargs.get('tool_calls'):
            tool_call = message.additional_kwargs['tool_calls'][0]
            print(f"[Tool Call: {tool_call['function']['name']}({tool_call['function']['arguments']})]")
    elif isinstance(message, ToolMessage):
        print(f"Tool Result: {message.content}")
    print()
