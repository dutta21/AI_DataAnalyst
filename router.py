"""
router.py
---------
Sits between app.py (Streamlit chat UI) and news_agent.py (NewsData.io fetch).

Uses Groq's OpenAI-compatible tool-calling to let the model itself decide,
per user turn, whether it needs live news data or can answer directly.

Flow:
    app.py -> router.route_query(client, messages) -> news_agent.get_news()  [if needed]
                                                     -> direct answer          [otherwise]
"""

import json
from news_agent import get_news

MODEL = "openai/gpt-oss-120b"

# Tool schema the model uses to request a news lookup.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": (
                "Fetch recent news articles for a topic. Use this when the user "
                "asks about current events, recent news, or anything that requires "
                "up-to-date information you wouldn't already know."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query / topic to fetch news for, e.g. 'Indian Economy'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of articles to fetch (max 10).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]


def _format_articles(articles):
    """Turn news_agent's article list into a compact string for the model to read."""
    if not articles:
        return "No articles found."

    lines = []
    for i, article in enumerate(articles, 1):
        title = article.get("title") or "Untitled"
        link = article.get("link") or ""
        desc = article.get("description") or ""
        lines.append(f"{i}. {title}\n   {desc}\n   Source: {link}")
    return "\n".join(lines)


def route_query(client, messages, temperature=0.2):
    """
    Given a Groq client and the full message history (system + prior turns +
    latest user message), decide whether to answer directly or fetch news
    first, and return the final assistant text.
    """

    # First pass: let the model see the tools and decide.
    first_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=temperature
    )

    choice = first_response.choices[0].message
    tool_calls = getattr(choice, "tool_calls", None)

    # No tool call -> model answered directly, we're done.
    if not tool_calls:
        return choice.content

    # Model wants live news. Execute each requested tool call and collect
    # the formatted results as plain text.
    #
    # NOTE: we deliberately do NOT replay the assistant's tool_calls message
    # or a "tool" role message back into the conversation. openai/gpt-oss-120b
    # on Groq appears to treat the presence of a prior tool_calls turn as a
    # standing cue to keep calling tools -- it kept doing so even with
    # tool_choice="none" explicitly set on the follow-up request, which Groq
    # then rejects as a 400 ("Tool choice is none, but model called a tool").
    # Instead, we inject the fetched news as plain context in a fresh
    # completion request that never declares `tools` at all, so the model
    # has no tool-calling shape to imitate.
    tool_results = []
    for tc in tool_calls:
        if tc.function.name == "get_news":
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            query = args.get("query", "")
            limit = args.get("limit", 5)

            try:
                articles = get_news(query, limit=limit)
                tool_results.append(_format_articles(articles))
            except Exception as e:
                tool_results.append(f"Error fetching news: {e}")
        else:
            tool_results.append(f"Unknown tool: {tc.function.name}")

    news_context = "\n\n".join(tool_results)

    followup_messages = messages + [
        {
            "role": "system",
            "content": (
                "Live news results were just fetched for the user's request:\n\n"
                f"{news_context}\n\n"
                "Use this information to answer the user's most recent message. "
                "Don't mention that a tool was called; just answer naturally, "
                "citing article titles/sources where relevant."
            )
        }
    ]

    # Second pass: plain completion, no tools declared at all.
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=followup_messages,
        temperature=temperature
    )

    return final_response.choices[0].message.content