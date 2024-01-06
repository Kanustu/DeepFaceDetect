def create_df(state):
    """
    Reads a CSV file corresponding to the specified state from the 'Metadata' directory
    and returns the data as a Pandas DataFrame.

    Parameters:
    - state (str): The name of the state for which the CSV file is to be read.

    Returns:
    - DataFrame: A Pandas DataFrame containing the data from the specified CSV file.
    """
    import pandas as pd

    var = pd.read_csv(f'../Metadata/{state}.csv')
    return var
