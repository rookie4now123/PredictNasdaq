import pandas as pd

def generate_features(df):
    df_new = pd.DataFrame(index=df.index)

    for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
        df_new[f'{col.lower()}_1'] = df[col].shift(1)

    df_new['avg_price_21'] = df['Close'].rolling(21).mean().shift(1)
    df_new['volatility_21'] = df['Close'].rolling(21).std().shift(1)
    df_new['return_1'] = df['Close'].pct_change().shift(1)

    df_new['close'] = df['Close']
    df_new.dropna(inplace=True)

    return df_new