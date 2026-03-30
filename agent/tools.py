from langchain.tools import tool

@tool
def predict_tool(query: str) -> str:
    """
    Predict stock price based on user query.
    Use this when the user asks about future price or forecast.
    """
    return f"Prediction requested for: {query}"


@tool
def sentiment_tool(text: str) -> str:
    """
    Analyze market sentiment from financial news or context.
    Use this when user asks about market mood or news impact.
    """
    return f"Sentiment analyzed for: {text}"