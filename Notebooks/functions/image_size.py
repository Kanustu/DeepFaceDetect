def path_list(data):
    """
    Extracts the 'path' column from a DataFrame and returns it as a list.

    Parameters:
    - data (DataFrame): The input DataFrame containing a 'path' column.

    Returns:
    - list: A list containing the values from the 'path' column.
    """
    return data['path'].to_list()


def image_size(path):
    """
    Retrieves the dimensions (width, height) of images given their file paths.

    Parameters:
    - path (list): A list of file paths to the images.

    Returns:
    - list: A list of tuples, where each tuple represents the dimensions of an image (width, height).
    """
    from PIL import Image
    
    size_list = []
    for x in range(len(path)):
        size_list.append(Image.open(f'../real_vs_fake/real-vs-fake/{path[x]}').size)
    
    return size_list
