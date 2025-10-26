# models/persistence.py
import joblib

def save_model(path, model, scaler=None, extras=None):
    """
    Save a model along with optional scaler and extra information.
    
    Args:
        path (str): File path to save the model.
        model: Trained model object.
        scaler: Optional scaler object.
        extras (dict): Any extra info to save with the model.
    """
    payload = {'model': model}
    if scaler is not None:
        payload['scaler'] = scaler
    if extras is not None:
        payload.update(extras)
    
    joblib.dump(payload, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    Load a model saved by save_model().
    
    Args:
        path (str): File path of the saved model.
    
    Returns:
        dict: Dictionary containing 'model', optional 'scaler', and extras.
    """
    return joblib.load(path)
