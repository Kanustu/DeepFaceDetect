def load_models(file_paths):
    """
    Load machine learning models from pickle files.

    Parameters:
    - file_paths (list): List of file paths to pickle files containing machine learning models.

    Returns:
    - list: List of loaded machine learning models.
    """
    models = []

    # Iterate through the list of file paths
    for file_path in file_paths:
        # Load each machine learning model from the pickle file
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
            models.append(model)

    return models
