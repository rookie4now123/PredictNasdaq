import torch

def save_model(model, scaler_X, scaler_y, path="models/lstm.pth"):
    torch.save({
        "model_state": model.state_dict(),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y
    }, path)


def load_model(model_class, input_size, path="models/lstm.pth"):
    checkpoint = torch.load(path)

    model = model_class(input_size)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint["scaler_X"], checkpoint["scaler_y"]