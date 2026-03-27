import yfinance as yf

def load_data():
    df = yf.download("^IXIC", start="1990-01-01")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df