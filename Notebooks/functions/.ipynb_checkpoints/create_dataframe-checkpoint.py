def create_df(state):
    import pandas as pd
    var =  pd.read_csv(f'./Metadata/{state}.csv')
    return var