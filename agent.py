from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from tools import (
    calculator,
    get_customer_debt,
    search_persian_docs,
    send_email,
    schedule_email,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)
# ============================================================
# Configuration
# ============================================================

TEHRAN_TZ = ZoneInfo("Asia/Tehran")

# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)

evaluator_llm = llm

# ============================================================
# Tools
# ============================================================

tools = [
    calculator,
    get_customer_debt,
    search_persian_docs,
    send_email,
    schedule_email,
]

llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# ============================================================
# Agent State
# ============================================================

class AgentState(MessagesState):
    retrieval_relevant: bool
    evaluation_reason: str
    rewritten_query: str
    retry_count: int
    email_sent: bool

# ============================================================
# LLM Node
# ============================================================
def call_llm(state: AgentState):

    now = datetime.now(TEHRAN_TZ)

    current_time = now.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )

    messages = state["messages"]

    system_prompt = f"""
            You are an AI assistant for a company.

            You can:

            1. Search Persian company documents.
            2. Send emails immediately.
            3. Schedule emails for future times.
            4. Perform calculations.

            Current date and time:
            {current_time}

            DOCUMENT SEARCH
            ---------------

            If the user asks for information that may exist in company documents,
            use `search_persian_docs`.

            Do not answer company-document questions from your own knowledge.

            EMAIL
            -----

            If the user asks to send an email immediately,
            use `send_email`.

            If the user asks to send an email in the future,
            use `schedule_email`.

            Do not merely describe an email action.
            Actually call the appropriate tool.

            EMAIL ADDRESS
            -------------

            If the user provides an email address,
            preserve it exactly.

            Never correct, modify, autocomplete,
            or guess an email address.

            DOCUMENTS + EMAIL
            -----------------

            If the user asks you to find information in company documents
            and then send that information by email:

            1. Search the documents first.
            2. Wait for the retrieval result.
            3. Use the retrieved information.
            4. Send the email.

            Do not send the email before document retrieval.

            RELATIVE TIME
            -------------

            For expressions such as:

            - one minute later
            - five minutes later
            - ten minutes later
            - two hours later

            calculate the requested future time relative to the current time.

            FOLLOW-UP REQUESTS
            ------------------

            If the user refers to a previous email request,
            reuse the previous recipient, subject and body
            unless the user explicitly changes them.

            IMPORTANT
            ---------

            The document search process is controlled by the workflow.

            If a document search has already happened,
            do not independently decide to search again unless
            the workflow explicitly asks for another search.

            After an email tool succeeds,
            provide a short confirmation.
            """

    messages_with_system = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + messages

    response = llm_with_tools.invoke(
        messages_with_system
    )

    return {
        "messages": [response]
    }

# ============================================================
# Retrieval Evaluation
# ============================================================

def evaluate_retrieval(state: AgentState):

    print("========== EVALUATOR CALLED ==========")

    messages = state["messages"]

    # --------------------------------------------------------
    # Original user question
    # --------------------------------------------------------

    user_question = None

    for message in messages:

        if isinstance(message, HumanMessage):
            user_question = message.content
            break

        if getattr(message, "type", None) == "human":
            user_question = message.content
            break

    # --------------------------------------------------------
    # Latest retrieval result
    # --------------------------------------------------------

    tool_result = None

    for message in reversed(messages):

        if getattr(message, "type", None) == "tool":

            tool_result = message.content
            break

    if not user_question:

        return {
            "retrieval_relevant": False,
            "evaluation_reason": (
                "The original user question is missing."
            ),
        }

    if not tool_result:

        return {
            "retrieval_relevant": False,
            "evaluation_reason": (
                "The retrieval result is missing."
            ),
        }

    print("ORIGINAL QUESTION:")
    print(user_question)
    print()

    # ========================================================
    # Evaluator
    # ========================================================

    evaluator_prompt = f"""
            You are evaluating the result of a Persian company-document retrieval system.

            Your job is ONLY to determine whether the retrieved documents contain
            enough information to answer the INFORMATION part of the user's request.

            You must NOT evaluate email sending, scheduling, recipient, address,
            subject, or other actions.


            ORIGINAL USER REQUEST
            ---------------------

            {user_question}


            RETRIEVED DOCUMENTS
            -------------------

            {tool_result}


            IMPORTANT RULES
            ---------------

            1. Focus on the information requested by the user.

            2. A broad user question does NOT require an exact wording match.

            3. Related grammatical forms count as the same concept.

            For example:

            "انبار"

            and

            "انباردار"

            and

            "وظایف انباردار"

            may refer to the same warehouse-related topic
            when the retrieved content clearly describes warehousekeeper duties.

            4. If the user asks:

            "انبار کارش چیه؟"

            and the retrieved document contains statements such as:

            "انباردار موظف است..."

            then this is RELEVANT because the document provides information
            about what the warehousekeeper does.

            5. Do NOT require the user to explicitly say "مسئولیت" or "وظایف".

            Questions such as:

            "انبار کارش چیه؟"
            "انباردار چه کارهایی انجام می‌دهد؟"
            "وظایف انباردار چیست؟"

            can refer to the same information.

            6. Different roles must remain different.

            For example:

            "انباردار"

            is not automatically the same as:

            "مدیر انبار"

            7. If the retrieved document contains concrete information that can
            reasonably answer the user's information request, return YES.

            8. Return NO only when the retrieved information genuinely does not
            provide enough information to answer the requested topic.


            DECISION FORMAT
            ---------------

            Return exactly two lines.

            Line 1:

            YES

            or:

            NO

            Line 2:

            A short explanation.
            """

    response = evaluator_llm.invoke(
        evaluator_prompt
    )

    result = response.content.strip()

    print("========== EVALUATOR RESULT ==========")
    print(result)
    print("=======================================")

    lines = result.splitlines()

    if not lines:

        return {
            "retrieval_relevant": False,
            "evaluation_reason": (
                "The evaluator returned an empty response."
            ),
        }

    decision = lines[0].strip().upper()

    reason = "\n".join(
        lines[1:]
    ).strip()

   
    decision = lines[0].strip().upper()
    reason = "\n".join(lines[1:]).strip()

    return {
        "retrieval_relevant": decision == "YES",
        "evaluation_reason": reason,
    }

# ============================================================
# Query Rewriting
# ============================================================

def rewrite_query(state: AgentState):

    messages = state["messages"]

    user_question = None

    for message in messages:

        if isinstance(message, HumanMessage):
            user_question = message.content
            break

        if getattr(message, "type", None) == "human":
            user_question = message.content
            break

    evaluation_reason = state.get(
        "evaluation_reason",
        "",
    )

    rewrite_prompt = f"""
            You are improving a search query for a Persian company-document
            retrieval system.

            Original user request:

            {user_question}

            Reason the previous retrieval was considered insufficient:

            {evaluation_reason}

            Create ONE short search query containing ONLY the information
            the user wants to find in the company documents.

            Do NOT include:

            - email
            - email address
            - recipient
            - scheduling
            - sending time
            - subject
            - email body
            - actions unrelated to document retrieval


            IMPORTANT:
            Do not use chineese language
            Do not change the meaning of the user's question.

            Do not change:

            - person
            - role
            - department
            - product
            - process
            - concept

            Examples:

            Original:
            "انباردار چه وظایفی دارد و نتیجه را ایمیل کن"

            Query:
            "وظایف انباردار"

            Original:
            "الزامات تردد در سالن تولید را پیدا کن و به من ایمیل کن"

            Query:
            "الزامات تردد در سالن تولید"

            Original:
            "انبار کارش چیه؟"

            Query:
            "وظایف و فعالیت های انباردار"

            Return ONLY the search query.
            """

    response = llm.invoke(
        rewrite_prompt
    )

    rewritten_query = response.content.strip()

    print("========== QUERY REWRITE ==========")

    print("NEW QUERY:")
    print(rewritten_query)

    print("===================================")

    return {
        "rewritten_query": rewritten_query,
        "retry_count": 1,
    }

# ============================================================
# Direct Retrieval Again
# ============================================================

def retrieve_again(state: AgentState):

    rewritten_query = state.get(
        "rewritten_query"
    )

    if not rewritten_query:

        return {
            "retrieval_relevant": False,
            "evaluation_reason": (
                "No rewritten query was generated."
            ),
        }

    result = search_persian_docs.invoke(
        {
            "query": rewritten_query
        }
    )

    return {
        "messages": [result]
    }

# ============================================================
# Routing After Evaluation
# ============================================================

def route_after_evaluation(state: AgentState):

    # Retrieval is good.
    # Continue the original user request.
    if state.get("retrieval_relevant") is True:

        return "continue_action"

    # If retry has already happened,
    # do not retry infinitely.
    retry_count = state.get(
        "retry_count",
        0,
    )

    if retry_count >= 1:

        return "fallback"

    # First failure -> rewrite query.
    return "rewrite_query"

# ============================================================
# Retrieval Fallback
# ============================================================

def retrieval_fallback(state: AgentState):

    return {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "اطلاعات کافی برای پاسخ به این سؤال "
                    "در اسناد موجود پیدا نشد."
                ),
            }
        ],
        "rewritten_query": None,
    }

# ============================================================
# Continue After Successful Retrieval
# ============================================================

def continue_after_retrieval(state: AgentState):

    messages = state["messages"]

    user_request = None

    for message in messages:

        if isinstance(message, HumanMessage):

            user_request = message.content
            break

        if getattr(message, "type", None) == "human":

            user_request = message.content
            break

    continuation_message = f"""
            The document retrieval process has already completed successfully.

            The retrieved documents were evaluated as relevant.

            Original user request:

            {user_request}


            Continue and COMPLETE the original request.

            IMPORTANT:

            - Do NOT search the documents again.
            - The retrieval process is already finished.
            - If the user requested an email, call the appropriate email tool.
            - If the user only requested information, answer using the retrieved documents.
            - Do not ask the user to repeat the request.
            """

    action_llm = llm.bind_tools(
        [
            send_email,
            schedule_email,
        ]
    )

    response = action_llm.invoke(
        messages + [
            {
                "role": "user",
                "content": continuation_message,
            }
        ]
    )

    return {
        "messages": [response],
        "rewritten_query": None,
    }

# ============================================================
# Tool Routing
# ============================================================

def route_after_tools(state: AgentState):

    messages = state["messages"]

    for message in reversed(messages):

        if (
            getattr(message, "type", None) != "ai"
            or not message.tool_calls
        ):
            continue

        for tool_call in message.tool_calls:

            tool_name = tool_call["name"]

            if tool_name == "search_persian_docs":

                return "evaluate_retrieval"

            if tool_name in {
                "send_email",
                "schedule_email",
            }:

                return "email_completed"

        break

    return "llm"

# ============================================================
# LLM Routing
# ============================================================

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    if last_message.tool_calls:

        return "tools"

    return END


# ============================================================
# Graph
# ============================================================

graph_builder = StateGraph(AgentState)

# ------------------------------------------------------------
# Nodes
# ------------------------------------------------------------

graph_builder.add_node("llm", call_llm,)

graph_builder.add_node("tools", tool_node,)

graph_builder.add_node("evaluate_retrieval", evaluate_retrieval,)

graph_builder.add_node("rewrite_query", rewrite_query,)

graph_builder.add_node("retrieve_again", retrieve_again,)

graph_builder.add_node("continue_after_retrieval", continue_after_retrieval,)

graph_builder.add_node("retrieval_fallback", retrieval_fallback,)

# ============================================================
# Edges
# ============================================================

graph_builder.add_edge(START, "llm",)

# ------------------------------------------------------------
# LLM -> Tools / END
# ------------------------------------------------------------

graph_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

# ------------------------------------------------------------
# Tools -> Evaluation / Email / LLM
# ------------------------------------------------------------

graph_builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "evaluate_retrieval": "evaluate_retrieval",
        "email_completed": END,
        "llm": "llm",
    },
)

# ------------------------------------------------------------
# Evaluation -> Continue / Rewrite / Fallback
# ------------------------------------------------------------

graph_builder.add_conditional_edges(
    "evaluate_retrieval",
    route_after_evaluation,
    {
        "continue_action": "continue_after_retrieval",
        "rewrite_query": "rewrite_query",
        "fallback": "retrieval_fallback",
    },
)

# ------------------------------------------------------------
# Rewrite -> Direct Retrieval
# ------------------------------------------------------------

graph_builder.add_edge("rewrite_query", "retrieve_again",)

# ------------------------------------------------------------
# Second Retrieval -> Evaluation
# ------------------------------------------------------------

graph_builder.add_edge("retrieve_again", "evaluate_retrieval",)

# ------------------------------------------------------------
# Fallback -> END
# ------------------------------------------------------------

graph_builder.add_edge("retrieval_fallback", END,)

# ============================================================
# Compile
# ============================================================

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer,
)
