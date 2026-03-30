import yfinance as yf


def load_data(symbol="^IXIC"):
    df = yf.download(symbol, period="5y", interval="1d")
    return df.dropna()


def get_latest_price(symbol="^IXIC"):
    ticker = yf.Ticker(symbol)
    return ticker.history(period="1d")["Close"].iloc[-1]