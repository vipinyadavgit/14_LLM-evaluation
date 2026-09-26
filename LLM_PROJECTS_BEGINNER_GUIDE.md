# Beginner Guide to LLM Applications

This guide walks through the building blocks used in the AI Tutor, Finance chatbot, LangChain chatbot, RAG, LLM evaluation, and LangGraph projects. It is organized so that a beginner can start with one model call and add more structure only when the project needs it.

## 1. The Basic Picture

An LLM application usually has a few parts:

1. **User input**: A question or instruction from the user.
2. **Application logic**: Code that validates input, builds a prompt, calls a model, and handles the result.
3. **LLM provider**: A service such as Groq or OpenAI that runs the model.
4. **User interface**: A command line, web page, or API through which the user interacts with the application.
5. **Optional supporting data**: Conversation history, documents, test examples, or application state.

The simplest flow is:

```text
User question -> Build messages -> Call LLM -> Show answer
```

LangChain can organize the prompt and model call into reusable components. RAG adds document retrieval before generation. LangGraph adds explicit state, steps, and routing for workflows that have multiple stages.

## 2. Common Setup for Every Project

These setup practices apply to chatbots, RAG systems, evaluators, and graph-based agents.

### Step 1: Define the job of the application

Write down what the application should do, who will use it, what information it may use, and what it must not do. A narrow scope makes prompts, tests, and user expectations clearer.

### Step 2: Create a project and Python environment

Use one virtual environment per project so its installed packages do not interfere with other projects. With `uv`, a basic project can be initialized and run like this:

```bash
uv init
uv add groq python-dotenv
uv run python main.py
```

Add only the packages the project uses. For example, a Streamlit app also needs `streamlit`; a LangChain app may need `langchain-core` and `langchain-groq`; a PDF RAG project may need `pypdf`, `faiss-cpu`, and `sentence-transformers`.

When using `uv`, prefer declaring dependencies with `uv add package-name`. This updates `pyproject.toml` and `uv.lock`, keeping the environment reproducible. Avoid leaving the dependency list empty while relying on a separate requirements file; `uv run` uses the project metadata and lockfile to manage the project environment.

If using pip instead, list dependencies in `requirements.txt`, create and activate a virtual environment, and install them with `pip install -r requirements.txt`. Pick one clear workflow and document it.

### Step 3: Use a clear source layout

A small script project can start with:

```text
my_project/
  .env                 # Local secrets; do not commit
  .gitignore
  pyproject.toml
  uv.lock
  README.md
  main.py
  app/
    __init__.py
    config.py
    llm.py
    prompts.py
```

As a project grows, keep related responsibilities in separate modules. Use valid Python package names that start with a letter or underscore, for example `finance_chatbot`; avoid package names that start with digits. A distribution name in `pyproject.toml` can use hyphens, while its Python import package normally uses underscores.

### Step 4: Store secrets outside source code

Create a local `.env` file:

```dotenv
GROQ_API_KEY=put_your_key_here
MODEL_NAME=your_model_name
```

Load it from Python and validate required values before making requests:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing")
```

Add `.env`, `.venv/`, `__pycache__/`, build output, and generated files to `.gitignore`. A `.env.example` may be committed if it contains placeholder values only. If a real key has ever been committed or shared, revoke it with the provider and create a new one; deleting the file later does not erase it from Git history.

### Step 5: Separate configuration, prompts, model calls, and interface

- `config.py` loads settings and environment variables.
- `prompts.py` contains system instructions and prompt templates.
- `llm.py` creates the model client and wraps model calls.
- `app.py` or `main.py` accepts user input and displays results.
- Other modules contain focused business logic such as retrieval, evaluation, or graph nodes.

This separation helps you change the UI without rewriting the model call, or change the model without rewriting the workflow.

### Step 6: Handle errors and test small pieces

Check for empty input, missing configuration, network failures, and malformed model output. Start testing with small examples that do not require a real API call where possible. Keep a few representative questions and expected behavior as a repeatable manual or automated test set.

## 3. Build a Basic LLM Chatbot

This is the best starting point: one user question goes to one model and one answer comes back.

### Step 1: Create a model client

With a provider SDK such as Groq's:

```python
from groq import Groq

client = Groq(api_key=api_key)
```

Creating the client prepares an object for making requests. It does not generate an answer yet.

### Step 2: Build messages

Chat APIs commonly use messages with roles:

- `system`: The assistant's role, scope, and response rules.
- `user`: The current question.
- `assistant`: A previous model response, when including conversation history.

```python
messages = [
    {"role": "system", "content": "You are a concise AI tutor."},
    {"role": "user", "content": user_question},
]
```

### Step 3: Send the request and read the answer

```python
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.2,
)
answer = response.choices[0].message.content
```

This is the direct-provider style used in the AI Tutor and Groq projects. The provider's response object has provider-specific fields, so you extract the message content from that response.

### Step 4: Add a user interface

A command-line interface can use `input()` and `print()`. A Streamlit interface can use `st.text_input()` and a button. Keep the same function for obtaining an answer so the interface only handles input and display.

### Step 5: Preserve conversation history when needed

For a multi-turn chatbot, keep prior user and assistant messages in a list and send them with the next request. In Streamlit, `st.session_state` can preserve this list across page reruns. Avoid sending unrelated users' conversations together; each user's history should be isolated.

## 4. Build a Streamlit Chatbot

Streamlit makes a simple web UI from Python code.

1. Install Streamlit with `uv add streamlit`.
2. Create `app.py` and add a title, input control, and send button.
3. Store conversation messages in `st.session_state` so they survive reruns.
4. Call a separate LLM function after validating that the input is not blank.
5. Append the user's message and assistant's answer to the conversation history.
6. Display the history in order and show a friendly message if the API call fails.
7. Run it with `uv run streamlit run app.py`.

Put the model call behind the button's valid-input branch. Otherwise an empty submission can leave the answer variable unset and cause a later error. Show an error for the failed request and do not append a made-up answer to the chat.

## 5. Build a LangChain Application

LangChain provides common interfaces and composable pieces. It does not replace the model provider; it calls the provider through a LangChain integration.

### Direct SDK and LangChain comparison

Direct Groq SDK:

```python
response = client.chat.completions.create(...)
answer = response.choices[0].message.content
```

LangChain with Groq:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model=model_name,
    temperature=0.2,
    api_key=api_key,
)

response = llm.invoke("Explain what an LLM is.")
answer = response.content
```

`ChatGroq(...)` creates a configured model object. The request happens at `llm.invoke(...)`. The result is a LangChain message object; `.content` extracts its text in common text-only use cases.

### Compose a prompt, model, and parser

LangChain's Runnable interface lets components be connected with `|`:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor."),
    ("user", "{input}"),
])

chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"input": "What is an embedding?"})
```

The dictionary key `input` fills `{input}` in the prompt. The chain formats messages, invokes the model, then converts the result to a string. Keep reusable prompt and chain creation in their own modules when the project grows.

### Add chat history deliberately

A chain that only passes `{"input": question}` is a single-turn chain, even if its interface looks like a chatbot. To make it multi-turn, store prior messages and explicitly include them in the prompt or use LangChain's message-history components. Test that a follow-up question can use information from the earlier turn.

## 6. Build a RAG Application

RAG means **Retrieval-Augmented Generation**. Before asking the LLM to answer, the application searches its own documents and gives relevant excerpts to the model.

```text
One-time or repeatable ingestion:
PDFs -> Extract text -> Split into chunks -> Embed chunks -> Save search index

For each question:
Question -> Embed question -> Find relevant chunks -> Prompt with context -> LLM answer
```

### Ingestion steps

1. **Load documents**: Use an appropriate loader, such as `PyPDFLoader` for PDF files.
2. **Split into chunks**: Break long documents into smaller overlapping passages. For example, start with roughly 500 characters and 50 characters of overlap, then tune based on the document and model.
3. **Create embeddings**: Use the same embedding model for document chunks and user questions. An embedding is a numeric representation used to compare meaning.
4. **Build an index**: Store vectors in a search index such as FAISS. If using inner-product similarity as cosine similarity, normalize both document and query embeddings consistently.
5. **Save the index and source text**: Store enough information to map each result back to its text and source document. Regenerate the index if the source files or embedding model changes.

### Question-time retrieval and answer

1. Embed the user's question with the same embedding model.
2. Search the vector index for the top `k` matching chunks.
3. Give the question and retrieved text to the LLM in a prompt.
4. Instruct the model to answer only from that context and say when the context is insufficient.
5. Return the answer, ideally with source or page references so a user can verify it.

Use defensive checks: fail with a clear message if ingestion has not run, handle an empty document folder, and avoid returning invalid search results when the index has fewer than `k` chunks. Do not blindly trust retrieved text as instructions; it is reference data, not a replacement for system rules. Treat serialized files such as pickle files as trusted local artifacts only, because loading untrusted pickle data can execute code.

In the railway refund example, the PDF is the knowledge source. A good answer must reflect the retrieved policy passage; the LLM should not invent a refund rule just because it sounds plausible.

## 7. Build an LLM Evaluation Project

Evaluation checks how well a model performs on a repeatable set of examples.

### Basic evaluation workflow

1. **Create a golden dataset**: Each record has an ID, question, and reference answer. Store it in JSON or another structured format.
2. **Generate an answer**: Send each question to the model being tested.
3. **Judge the answer**: Compare the generated answer with the reference and ask a judge model for clearly defined scores.
4. **Validate the result**: Check that the judge returned all required fields and that scores are numeric and within the allowed range.
5. **Save results**: Store the question, reference, generated response, scores, model names, and run metadata in CSV or JSON for later analysis.
6. **Compare runs**: Use the same dataset when changing prompts or models so the results are comparable.

### Use an LLM judge carefully

Define each score so that the judge knows what a high or low score means. For example, a hallucination score must say whether 10 means "no hallucination" or "many hallucinations"; ambiguous direction makes results hard to interpret. Ask for structured JSON when supported, parse it, and reject invalid or incomplete outputs rather than silently accepting them.

LLM judges can be inconsistent or biased toward their own style. Use human review for important decisions, inspect examples behind score changes, and do not treat a score as objective truth. Track API cost and latency, since a two-model evaluation calls the generator and judge for each example.

## 8. Build a LangGraph Workflow or Agent

LangGraph is useful when a task has multiple named steps, shared state, branches, or loops. A graph is made of **state**, **nodes**, and **edges**.

- **State**: The structured data carried through the workflow, such as a requirement, generated code, feedback, and iteration count.
- **Node**: A function that reads state, does one job, and returns the fields it updates.
- **Edge**: A connection that says which node runs next.
- **Conditional edge**: A routing function that chooses the next node from the current state.
- **END**: A terminal point where the graph finishes.

### Step-by-step graph construction

1. **Write the workflow in plain language**. Example: generate code, review it, run QA, fix if required, then report.
2. **Design a typed state** with every value the workflow needs. For example, `requirement`, `generated_code`, `review_feedback`, `qa_feedback`, `needs_fix`, `iteration_count`, and `final_report`.
3. **Write one small function per node**. A developer node should generate code; a review node should review it; a QA node should decide whether it passes. Return only the fields being updated.
4. **Create the graph** with `StateGraph(AgentState)` and register the nodes by name.
5. **Set the entry point** to the first node.
6. **Connect fixed steps** with normal edges.
7. **Add conditional routing** for decisions, such as routing from QA to either fix or report.
8. **Bound loops** with an iteration counter and maximum. A workflow must have a clear exit even when QA never passes.
9. **Compile and invoke** the graph with an initial state, then inspect the final state.
10. **Test routes and limits**: test both pass and fail decisions, missing fields, and the maximum-iteration path.

Example software-development-agent flow:

```text
START -> Developer -> Reviewer -> QA
                              QA passes -> Report -> END
                              QA fails  -> Fix -> Reviewer
                                          (repeat only up to a limit)
```

Make routing values explicit and stable. For example, prefer a structured boolean such as `needs_fix` over searching free-form feedback for a phrase like `NEEDS_FIX = YES`. If the QA model produces that boolean, validate it before routing. A graph organizes the workflow; it does not make generated code safe to execute. Review code and run it only in an appropriately isolated environment.

## 9. A Practical Build Order

When starting a new project, use this order to keep each step understandable:

1. Make one model call from a small Python function.
2. Add configuration and secret handling.
3. Add input validation and useful errors.
4. Add a CLI or Streamlit interface.
5. Add conversation history if the application needs multiple turns.
6. Add LangChain when reusable prompts, model interfaces, or composed steps help.
7. Add RAG when answers need private or domain-specific documents.
8. Add an evaluation dataset before making many prompt or model changes.
9. Add LangGraph when the workflow has meaningful stages, branches, or bounded loops.
10. Add tests, logging, source references, and deployment configuration as the application becomes important to users.

Do not add every framework to every project. A direct model call is often the clearest solution for a one-step chatbot. Each additional layer should solve a real need.

## 10. Before Sharing or Deploying

- Confirm `.env` and real API keys are not in Git history; rotate any exposed key.
- Ensure `pyproject.toml` declares the dependencies used by the application and commit `uv.lock` for reproducible `uv` environments.
- Check that the documented command works from the documented directory.
- Keep package names valid for Python imports and configure script entry points to a real function.
- Test empty input, missing keys, provider errors, invalid model output, and any graph branch or loop.
- Avoid logging API keys, private user data, or entire sensitive prompts.
- For medical, financial, or other high-impact topics, clearly limit the assistant's role and direct users to qualified professionals when appropriate.
- Tell users when an answer is based on retrieved documents, and provide sources where possible.
- Track model choice, prompt version, and evaluation dataset version so behavior changes can be understood.

## Glossary

- **API key**: A secret credential that authorizes requests to an LLM provider.
- **Prompt**: The instructions and input sent to a model.
- **Model client**: A configured object that sends requests to a provider.
- **Temperature**: A generation setting that influences variation. Lower values are generally more consistent, but do not guarantee correctness.
- **Embedding**: A vector representation of text used for semantic search.
- **Vector database/index**: A system for storing vectors and finding similar ones.
- **Chain**: A composition of prompt, model, parser, or other runnable steps.
- **Agent/workflow graph**: A stateful set of steps and routing rules that control what happens next.
- **Golden answer**: A reference response used to compare or evaluate generated answers.