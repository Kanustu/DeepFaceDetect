def unpickled(filename):
    import pickle
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model