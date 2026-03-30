import streamlit as st
import os

st.set_page_config(layout="wide", page_title="🤖 AI Quant Agent")
st.sidebar.title("🔐 API Configuration")
user_api_key = st.sidebar.text_input("Enter Your API Key", type="password")
if user_api_key:
    os.environ["GOOGLE_API_KEY"] = user_api_key
if not user_api_key:
    st.warning("Please enter your Google API key to continue.")
    st.stop()
import matplotlib.pyplot as plt
from data.loader import load_data, get_latest_price
from features.engineering import generate_features
from models.lstm_model import train_model, NasdaqLSTM
from models.utils import save_model, load_model
from rag.retriever import build_index
from rag.sentiment import analyze_sentiment
from agent.tools import predict_tool, sentiment_tool
from agent.orchestrator import build_agent
from evaluation.backtest import compute_returns, sharpe_ratio



# -------- Load Data --------
df = load_data()
features = generate_features(df)

X = features.drop("close", axis=1).values
y = features["close"].values

# -------- Model Handling --------
MODEL_PATH = "models/lstm.pth"

try:
    model, scaler_X, scaler_y = load_model(NasdaqLSTM, X.shape[1], MODEL_PATH)
    st.success("Loaded saved model")
except:
    st.warning("Training new model...")
    model, scaler_X, scaler_y = train_model(X, y)
    save_model(model, scaler_X, scaler_y)
    st.success("Model trained & saved")

# -------- Real-time price --------
latest_price = get_latest_price()
st.metric("📈 Live Price", f"{latest_price:.2f}")

# -------- RAG --------
query_engine = build_index()

@st.cache_resource
def get_agent(user_api_key):
    return build_agent(
        tools=[predict_tool, sentiment_tool],
        user_api_key=user_api_key
    )
agent = get_agent(user_api_key)

# -------- Chat UI --------
st.subheader("💬 Chat with AI Agent")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask something about the market:")

if user_input:
    context = str(query_engine.query(user_input))
    sentiment = analyze_sentiment(context)

    result = agent.invoke({"input": user_input})
    response = result["output"]

    st.session_state.chat_history.append((user_input, response))

for q, a in st.session_state.chat_history:
    st.write(f"🧑: {q}")
    st.write(f"🤖: {a}")

# -------- Prediction --------
st.subheader("📊 Backtesting")

preds = y[-100:]
actuals = y[-100:]

returns = compute_returns(preds, actuals)
sr = sharpe_ratio(returns)

st.metric("Sharpe Ratio", f"{sr:.2f}")

# Plot
fig, ax = plt.subplots()
ax.plot(preds, label="Predicted")
ax.plot(actuals, label="Actual")
ax.legend()

st.pyplot(fig)