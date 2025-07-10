from langgraph.graph import StateGraph
from langchain_ollama import OllamaLLM
from typing import TypedDict, Literal

# ✅ 1. Define state class
class AgentState(TypedDict):
    input_text: str
    result: str
    next: Literal["math", "summarizer", "fallback", "explain"]

# ✅ 2. Load model
llm = OllamaLLM(model="mistral")

# ✅ 3. Router logic
def router_node(state: AgentState) -> dict:
    prompt = state["input_text"].lower()
    if "summarize" in prompt:
        return {"next": "summarizer"}
    elif any(op in prompt for op in ["+", "-", "*", "/"]):
        return {"next": "math"}
    elif "explain" in prompt:
        return {"next": "explain"}
    else:
        return {"next": "fallback"}

# ✅ 4. Math Node
def math_node(state: AgentState) -> AgentState:
    expr = state["input_text"]
    response = llm.invoke(f"Solve this: {expr}")
    return {"input_text": expr, "result": response, "next": "printer"}

# ✅ 5. Summary Node
def summarizer_node(state: AgentState) -> AgentState:
    text = state["input_text"].replace("summarize:", "").strip()
    response = llm.invoke(f"Summarize this in simple terms: {text}")
    return {"input_text": state["input_text"], "result": response, "next": "printer"}

# ✅ 6. Fallback Node
def fallback_node(state: AgentState) -> AgentState:
    response = llm.invoke(f"Respond helpfully to this query: {state['input_text']}")
    return {"input_text": state["input_text"], "result": response, "next": "printer"}

# ✅ 7. Explain Concept Node
def explain_concept_node(state: AgentState) -> AgentState:
    text = state["input_text"].replace("explain:", "").strip()
    response = llm.invoke(f"Explain this in a simple and child-friendly way (age under 18): {text}")
    return {"input_text": state["input_text"], "result": response, "next": "printer"}

# ✅ 8. Printer Node
def printer_node(state: AgentState) -> AgentState:
    print("\n✅ Output:", state["result"])
    return state

# ✅ 9. Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("math", math_node)
graph.add_node("summarizer", summarizer_node)
graph.add_node("fallback", fallback_node)
graph.add_node("explain", explain_concept_node)
graph.add_node("printer", printer_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda x: x["next"],
    {
        "math": "math",
        "summarizer": "summarizer",
        "fallback": "fallback",
        "explain": "explain"
    }
)

graph.add_edge("math", "printer")
graph.add_edge("summarizer", "printer")
graph.add_edge("fallback", "printer")
graph.add_edge("explain", "printer")

app = graph.compile()

# ✅ 10. Dynamic Input Loop WITHOUT Exit Option
def run_interactive():
    print("\n🤖 Welcome to the LangGraph Agent Assistant!\n")
    while True:
        print("\n🔘 Choose an option:")
        print("1. Summary\n2. Math\n3. Fallback\n4. Explain Concept")
        choice = input("Enter your choice (1-4): ").strip()

        prompt_map = {
            "1": "summarize: ",
            "2": "",
            "3": "",
            "4": "explain: "
        }

        if choice not in prompt_map:
            print("❌ Invalid choice. Try again.")
            continue

        user_input = input("📝 Enter your input : ",).strip()
        full_input = prompt_map[choice] + user_input
        app.invoke({"input_text": full_input})

        cont = input("\n🔁 Do you want to continue? (yes/no): ").strip().lower()
        if cont not in ["yes", "y"]:
            print("\n👋 Exiting. Thank you!")
            break

# ✅ Entry Point
if __name__ == "__main__":
    run_interactive()

