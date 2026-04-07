from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from tools import search_flights, search_hotels, calculate_budget
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

# 1. Đọc System Prompt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# 2. Khai báo State
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 3. Khởi tạo LLM và Tools
tools_list = [search_flights, search_hotels, calculate_budget]

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    openai_api_base="https://openrouter.ai/api/v1", # URL của OpenRouter
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://your-app-url.com", # Bắt buộc với OpenRouter
        "X-Title": "Your App Name"
    }
)

llm_with_tools = llm.bind_tools(tools_list)


# 4. Agent Node
def agent_node(state: AgentState):
    messages = state["messages"]
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        for tc in response.tool_calls:
            # Truy cập bằng key thay vì thuộc tính chấm
            name = tc.get("name")
            args = tc.get("args")
            print(f"Tool called: {name} with args {args}")
    else:
        print("No tools called.")
    return {"messages": [response]}


# 5. Khởi tạo Graph (Cần thiết để chạy agent)
builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools_list))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("TravelBuddy – Trợ lý Du lịch Thông minh")
    print("   Gõ 'quit' để thoát")
    print("=" * 60)

    # Khởi tạo danh sách lịch sử chat
    chat_history = []

    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Tạm biệt!")
            break

        print("\nTravelBuddy đang suy nghĩ...")

        # 1. Ghi log USER vào file (sử dụng mode 'a' để ghi nối tiếp)
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"USER: {user_input}\n")

        # Thêm input của người dùng vào lịch sử
        chat_history.append(("human", user_input))

        # Gọi graph
        result = graph.invoke({"messages": chat_history})

        # Cập nhật lịch sử
        final = result["messages"][-1]
        chat_history.append(final)

        # 2. Xử lý nội dung để in ra màn hình và ghi log AGENT
        content = getattr(final, "content", str(final))

        # Nếu content trống (ví dụ khi gọi tool), ghi chú rõ là đang gọi tool
        if not content and hasattr(final, "tool_calls") and final.tool_calls:
            content = f"Đang thực hiện tool: {final.tool_calls[0]['name']}"

        print(f"\nTravelBuddy: {content}")

        # 3. Ghi log AGENT vào file
        with open("logs.txt", "a", encoding="utf-8") as f:
            f.write(f"AGENT: {content}\n")