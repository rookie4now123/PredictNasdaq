import numpy as np

def compute_returns(preds, actuals):
    returns = []

    for i in range(1, len(preds)):
        if preds[i] > preds[i-1]:
            returns.append((actuals[i] - actuals[i-1]) / actuals[i-1])
        else:
            returns.append(0)

    return np.array(returns)


def sharpe_ratio(returns, risk_free=0.01):
    excess = returns - risk_free / 252
    return np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)