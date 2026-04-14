from stable_baselines3 import PPO, SAC


def load_model(path, env=None, device="auto"):
    """
    Load a Stable Baselines3 model from a path, automatically detecting
    whether it is a PPO or SAC model.

    :param path: Path to the model zip file
    :param env: Optional environment to attach to the model
    :param device: Device to load the model on
    :return: The loaded model instance
    """
    # Attempt to load as PPO
    try:
        # We use a small trick: PPO and SAC have different internal structures.
        # SB3 usually fails early or during policy reconstruction if the class is wrong.
        return PPO.load(path, env=env, device=device)
    except Exception:
        pass

    # Attempt to load as SAC
    try:
        return SAC.load(path, env=env, device=device)
    except Exception as e:
        raise ValueError(
            f"Failed to load model from {path}. "
            f"The file may be corrupted or use an unsupported algorithm. "
            f"Error: {e}"
        )


def get_model_algo_name(model) -> str:
    """Return the name of the algorithm (e.g., 'PPO', 'SAC')."""
    return model.__class__.__name__
