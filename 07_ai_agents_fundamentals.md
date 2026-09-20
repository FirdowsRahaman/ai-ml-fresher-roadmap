# Module 07: AI Agents Fundamentals

This module covers the core concepts, architectures, and interview questions related to **Autonomous AI Agents**. Freshers are increasingly expected to understand how agents differ from static LLMs and how they plan, reason, and act in real-world environments.

---

## 🤖 Section 1: What is an AI Agent?

### Q1: What is the fundamental difference between an LLM, a Chain, and an AI Agent?
**Answer:**
| System | Definition | Decision Control | Example |
|---|---|---|---|
| **LLM** | A probabilistic next-token generator. | No control; one-shot output. | Generating a poem via ChatGPT. |
| **Chain** | A deterministic, hard-coded sequence of calls (e.g., Prompt A → Model → Parse → Prompt B). | Developer defines exact path; zero autonomy. | Standard RAG pipeline. |
| **AI Agent** | An autonomous system that uses an LLM to decide its *own* execution path, choose tools, and repeat steps until a goal is met. | Model dynamically decides steps, loops, and termination. | An agent that searches the web, tests Python code, fixes errors, and emails a report. |

```
Chain:  [Input] ────────► [Step 1] ────────► [Step 2] ────────► [Output]
                               (Hard-coded path)

Agent:  [Goal]  ────────► [Think] ◄───┐
                            │         │ (Autonomous Loop)
                            ▼         │
                         [Action] ────┘
                            │ (When goal achieved)
                            ▼
                         [Finish]
```

---

### Q2: What are the core components of an AI Agent architecture?
**Answer:**
A production AI Agent consists of four main pillars:
1. **Brain (Core LLM):** Performs reasoning, intent classification, and task evaluation.
2. **Planning & Reasoning:** Breaks down high-level user goals into sub-tasks (e.g., ReAct, Plan-and-Solve).
3. **Memory:**
   - *Short-Term Memory:* In-context conversation history and scratchpad of recent thoughts.
   - *Long-Term Memory:* External vector database or key-value store for cross-session recall.
4. **Tools:** External capabilities (APIs, databases, Python code interpreters, search engines) that let the agent affect the outside world.

---

## 🔄 Section 2: The ReAct Framework & The Agent Loop

### Q3: What is the ReAct framework, and why is it superior to pure reasoning or pure acting?
**Answer:**
**ReAct (Reasoning + Acting)** was introduced by Yao et al. (2022). It interleaves explicit reasoning steps (*"Thought"*) with action execution (*"Action"*) and feedback interpretation (*"Observation"*).
- **Pure Reasoning (Chain-of-Thought):** The model hallucinates facts because it cannot access external, real-time ground truth.
- **Pure Acting:** The model blindly triggers tools without reflecting on intermediate results, leading to catastrophic failure on complex multi-step tasks.
- **ReAct Advantage:** The model explains *why* it is selecting a tool, runs the tool, reads the real result, and course-corrects dynamically.

---

### Q4: How do you implement a minimal ReAct agent loop from scratch in pure Python?
**Answer:**
Here is an interview-ready, zero-dependency implementation of a ReAct agent execution loop:

```python
import re

# Mock tools
def search_database(query: str) -> str:
    mock_db = {"iPhone 16": "$799", "MacBook Pro": "$1599"}
    return mock_db.get(query, "Product not found")

def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "search": search_database,
    "calculate": calculate
}

def mock_llm_reasoning_step(prompt: str, iteration: int) -> str:
    """Simulates an LLM producing Thought, Action, and Action Input."""
    if iteration == 0:
        return (
            "Thought: I need to find the price of iPhone 16 first.\n"
            "Action: search\n"
            "Action Input: iPhone 16"
        )
    elif iteration == 1:
        return (
            "Thought: iPhone 16 costs $799. Now I need to calculate total for 3 units.\n"
            "Action: calculate\n"
            "Action Input: 799 * 3"
        )
    else:
        return (
            "Thought: I have the final total.\n"
            "Final Answer: The total cost for 3 iPhone 16 units is $2397."
        )

# Agent execution loop
def run_agent(user_goal: str, max_steps: int = 5):
    scratchpad = f"User Goal: {user_goal}\n"
    
    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        response = mock_llm_reasoning_step(scratchpad, iteration=step)
        print(response)
        
        if "Final Answer:" in response:
            final_ans = response.split("Final Answer:")[1].strip()
            return final_ans
        
        # Parse Action and Action Input
        action_match = re.search(r"Action:\s*(\w+)", response)
        input_match = re.search(r"Action Input:\s*(.+)", response)
        
        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()
            
            tool_fn = TOOLS.get(tool_name)
            observation = tool_fn(tool_input) if tool_fn else "Unknown tool"
            print(f"Observation: {observation}")
            
            scratchpad += f"\n{response}\nObservation: {observation}"
            
    return "Error: Agent reached maximum step limit without concluding."

result = run_agent("What is the cost of buying 3 iPhone 16 phones?")
print(f"\nResult: {result}")
```

---

## 🧠 Section 3: Memory & State Management

### Q5: How is Memory structured in production agents?
**Answer:**
1. **Working Memory (Scratchpad):** The current trajectory stored directly inside the LLM context window. Includes previous Thoughts, Actions, and Observations for the active task.
2. **Short-Term Conversational Memory:** Summary buffer or sliding window of chat turns between the user and assistant.
3. **Long-Term Episodic Memory:** Stores past interactions, successful workflows, and user preferences in an external Vector DB. Retrieved via semantic similarity when a similar task is initiated.
4. **Entity Memory:** A structured knowledge graph (e.g., Neo4j) maintaining facts about specific entities (e.g., "User prefers Python over JavaScript").

---

## 🗺️ Section 4: Planning, Reflection & Edge Cases

### Q6: What is the difference between Single-Path and Multi-Path Planning?
**Answer:**
- **Single-Path (ReAct, Step-by-Step):** The agent takes one action, waits for the result, and decides the next step sequentially. Simple, but prone to getting stuck in loops.
- **Plan-and-Solve:** The agent first creates a complete multi-step checklist (1, 2, 3...), then executes each sub-task sequentially, updating the plan if errors occur.
- **Tree-of-Thoughts (ToT) / Multi-Path:** The agent explores multiple alternative branches of reasoning, evaluates their potential outcomes using self-reflection, and backtracks if a path hits a dead end.

---

### Q7: What are the common failure modes of AI Agents, and how do you mitigate them?
**Answer:**
1. **Infinite Loops / Tool Thrashing:**
   - *Problem:* Agent calls the same tool repeatedly with identical parameters because it misinterprets the observation.
   - *Mitigation:* Set a hard `max_iterations` cutoff and implement state deduplication checks.
2. **Context Window Saturation:**
   - *Problem:* Complex loops produce massive observations (e.g., whole web pages), exhausting token limits.
   - *Mitigation:* Summarize tool outputs before appending them to the working memory scratchpad.
3. **Hallucinated Tools / Tool Schema Drift:**
   - *Problem:* Model invents non-existent functions or produces invalid JSON arguments.
   - *Mitigation:* Use strict JSON-Schema validation and return validation error strings directly into the agent's next observation loop for self-correction.

---

## 🚀 Key Takeaways for Interviews
- An agent differs from an LLM by maintaining an **autonomous observe-reason-act loop**.
- **ReAct** combines thought and action to prevent both pure hallucinations and unguided tool calls.
- Production agents require strict limits on iterations, token budgets, and structured memory layers.

---

[← Previous: Module 06 - GenAI, LLMs & Advanced RAG](./06_genai_llms_and_rag.md) | [Next: Module 08 - Tools & Orchestration →](./08_tools_and_orchestration.md)
