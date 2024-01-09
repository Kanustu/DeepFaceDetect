def color_info(path):
    """
    Extracts color information (bands) from a list of image files.

    Parameters:
    - path (list): List of file paths to image files.

    Returns:
    - list: A list containing color information (bands) for each image in the input list.
    """
    from PIL import Image

    # Initialize an empty list to store color information for each image
    color_list = []

    # Iterate through the list of image file paths
    for x in range(len(path)):
        # Open each image and retrieve its color information (bands)
        color_list.append(Image.open(f'../real_vs_fake/real-vs-fake/{path[x]}').getbands())

    return color_list



def color_test(color_info_list):
    """
    Filters a list of color information to include only RGB images.

    Parameters:
    - color_info_list (list): List of color information for images.

    Returns:
    - list: A filtered list containing only color information for RGB images (R, G, B).
    """
    color_list = []

    # Iterate through the list of color information for images
    for x in color_info_list:
        # Check if the color information corresponds to an RGB image (R, G, B)
        if x == ('R', 'G', 'B'):
            color_list.append(x)

    return color_list


def length_compare(var1, var2):
    """
    Compares the lengths of two variables.

    Parameters:
    - var1: First variable for length comparison.
    - var2: Second variable for length comparison.

    Returns:
    - bool: True if the lengths of var1 and var2 are equal, False otherwise.
    """
    if len(var1) == len(var2):
        return True
    else:
        return False
