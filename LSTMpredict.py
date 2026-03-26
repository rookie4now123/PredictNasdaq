import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Advanced AI/ML Imports ---
from transformers import pipeline
from llama_index.core import Document, VectorStoreIndex
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# ==========================================
# 1. DATA PREPARATION (Technical Indicators)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
file_name = 'nasdaq_historical_data.csv'
file_path = os.path.join(script_dir, file_name)

# Mock data creation if file doesn't exist for testing purposes
if not os.path.exists(file_path):
    dates = pd.date_range(start='1990-01-02', end='2026-03-24', freq='B')
    data_raw = pd.DataFrame(np.random.rand(len(dates), 5) * 100 + 5000, 
                            columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=dates)
    data_raw.index.name = 'Date'
else:
    data_raw = pd.read_csv(file_path, index_col='Date', parse_dates=True)

def generate_features(df):
    df_new = pd.DataFrame(index=df.index)
    # Original features
    for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
        df_new[f'{col.lower()}_1'] = df[col].shift(1)
    
    # Averages & Volatility (Rolling standard deviation used later for Agentic routing)
    df_new['avg_price_21'] = df['Close'].rolling(21).mean().shift(1)
    df_new['volatility_21'] = df['Close'].rolling(21).std().shift(1) # Crucial for LangChain Agent
    
    # Returns
    df_new['return_1'] = df['Close'].pct_change().shift(1)
    
    df_new['close'] = df['Close']
    df_new = df_new.dropna()
    return df_new

data = generate_features(data_raw)

start_train, end_train = '1990-01-02', '2025-10-27'
start_test, end_test = '2025-10-28', '2026-03-24'

data_train = data.loc[start_train:end_train]
data_test = data.loc[start_test:end_test]

# ==========================================
# 2. PYTORCH LSTM MODEL (Technical Predictor)
# ==========================================
class NasdaqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(NasdaqLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last time step
        return out

def create_sequences(X, y, seq_length=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

# Scaling
X_train_df = data_train.drop('close', axis=1)
y_train_df = data_train['close']
X_test_df = data_test.drop('close', axis=1)
y_test_df = data_test['close']

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled_train = scaler_X.fit_transform(X_train_df)
y_scaled_train = scaler_y.fit_transform(y_train_df.values.reshape(-1, 1))
X_scaled_test = scaler_X.transform(X_test_df)
y_scaled_test = scaler_y.transform(y_test_df.values.reshape(-1, 1))

seq_length = 5
X_seq_train, y_seq_train = create_sequences(X_scaled_train, y_scaled_train, seq_length)
X_seq_test, y_seq_test = create_sequences(X_scaled_test, y_scaled_test, seq_length)

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_seq_train)
y_train_t = torch.FloatTensor(y_seq_train)
X_test_t = torch.FloatTensor(X_seq_test)

# Train LSTM (Simulated quick training)
lstm_model = NasdaqLSTM(input_size=X_train_t.shape[2])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

print("Training PyTorch LSTM...")
for epoch in range(50): # Kept low for execution speed
    lstm_model.train()
    optimizer.zero_grad()
    outputs = lstm_model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

lstm_model.eval()
with torch.no_grad():
    lstm_predictions_scaled = lstm_model(X_test_t).numpy()
lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled).flatten()

# ==========================================
# 3. LLAMA-INDEX RAG & HUGGINGFACE SENTIMENT
# ==========================================
print("Initializing RAG and Sentiment Pipelines...")

# Hugging Face FinBERT for financial sentiment
# (Using default distilbert here for speed without downloading massive weights, 
# but in production, replace with "ProsusAI/finbert")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbase/sst2") 

# Mock RAG Database (SEC filings and News)
mock_news_data = [
    Document(text="Federal reserve indicates potential rate cuts, markets optimistic."),
    Document(text="Tech earnings beat expectations, driving Nasdaq heavily upward."),
    Document(text="Geopolitical tensions rise, causing uncertainty in global supply chains.")
]
rag_index = VectorStoreIndex.from_documents(mock_news_data)
query_engine = rag_index.as_query_engine()

def get_fundamental_sentiment(date_str):
    """Retrieves news for a given date via RAG, and analyzes sentiment."""
    # 1. RAG Retrieval
    query_response = query_engine.query(f"What is the market outlook around {date_str}?")
    context = str(query_response)
    
    # 2. Sentiment Grounding
    sentiment = sentiment_analyzer(context[:512])[0]
    
    # Convert sentiment to a numerical multiplier (e.g., POSITIVE = 1.02, NEGATIVE = 0.98)
    multiplier = 1.01 if sentiment['label'] == 'POSITIVE' else 0.99
    return multiplier

# ==========================================
# 4. LANGCHAIN AGENTIC ORCHESTRATOR
# ==========================================
# Instead of burning OpenAI tokens, we implement the Agent's *Logic* here:
# "Decides whether to prioritize technical indicators (ML) or fundamental news (RAG) based on market conditions."

def agentic_orchestrator(date, technical_pred, current_volatility, baseline_volatility):
    """
    Simulates Langchain Agent Routing:
    If market volatility is exceptionally high, technicals fail. The agent routes 
    to RAG/Fundamental Sentiment to adjust the prediction.
    """
    # Tool 1: LSTM Predictor (Already executed: technical_pred)
    # Tool 2: RAG Sentiment Analyzer
    
    # Market Condition Logic:
    is_high_volatility = current_volatility > (baseline_volatility * 1.5)
    
    if is_high_volatility:
        # Agent decides to use RAG Sentiment to ground the technical prediction
        sentiment_multiplier = get_fundamental_sentiment(date)
        final_prediction = technical_pred * sentiment_multiplier
        action_taken = "RAG Sentiment Adjusted"
    else:
        # Agent trusts the LSTM technical indicators
        final_prediction = technical_pred
        action_taken = "Trusted LSTM"
        
    return final_prediction, action_taken

# ==========================================
# 5. EXECUTE PIPELINE ON TEST DATA
# ==========================================
print("Running Agentic Orchestration on Test Data...")
final_predictions = []
actions = []

# Align dates with sequences (we lost `seq_length` rows at the start)
test_dates = data_test.index[seq_length:]
test_volatilities = data_test['volatility_21'].values[seq_length:]
baseline_vol = data_train['volatility_21'].mean()

for i in range(len(lstm_predictions)):
    date = test_dates[i].strftime('%Y-%m-%d')
    tech_pred = lstm_predictions[i]
    current_vol = test_volatilities[i]
    
    # Agent orchestrates the final prediction
    final_pred, action = agentic_orchestrator(date, tech_pred, current_vol, baseline_vol)
    
    final_predictions.append(final_pred)
    actions.append(action)

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 6))
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# True values (trimming the first `seq_length` to match predictions)
y_test_plot = y_test_df.values[seq_length:]

plt.plot(test_dates, y_test_plot, c='black', label='Ground Truth', linewidth=2)
plt.plot(test_dates, lstm_predictions, c='blue', alpha=0.5, label='Raw LSTM (Technicals)')
plt.plot(test_dates, final_predictions, c='red', linestyle='--', label='Agent-Orchestrated (LSTM + RAG)')

plt.title('Nasdaq Forecasting: PyTorch LSTM + RAG Sentiment Orchestration')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.xticks(rotation=45)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print a summary of Agent actions
action_df = pd.Series(actions).value_counts()
print("\n--- Agent Workflow Summary ---")
print(action_df)