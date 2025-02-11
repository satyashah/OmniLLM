from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from dotenv import load_dotenv

load_dotenv()

def add(a: int, b: int) -> int:
    """Adds a and b.


    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# Define LLM with bound tools
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

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
result = graph.invoke({"messages": [HumanMessage(content="What is (2 + 2) / 4 * 3?")]})

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
