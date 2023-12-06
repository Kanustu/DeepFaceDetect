def pickled(model, filename):
    import pickle
    return pickle.dump(model, open(filename, 'wb'))
    