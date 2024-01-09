def sample_images(state, num_samples):
    """
    Sample images from the 'fake' and 'real' classes in the specified metadata file.

    Reads the metadata from the './Metadata/{state}.csv' file and selects
    three random samples from the 'fake' class and three random samples from the 'real'
    class. Returns the paths to the sampled fake and real images.

    Args:
    - state (str): The state for which to sample images. Should be 'train', 'valid', or 'test'.

    Returns:
    - fake_images (list): List of paths to three sampled fake images.
    - real_images (list): List of paths to three sampled real images.
    """
    import pandas as pd

    # Read the metadata for the specified state
    meta = pd.read_csv(f'../metadata/{state}.csv')

    # Select paths for 'fake' and 'real' classes
    fake = meta[meta['label_str'] == 'fake']['path']
    real = meta[meta['label_str'] == 'real']['path']

    # Sample three images from each class
    fake_images = fake.sample(num_samples).tolist()
    real_images = real.sample(num_samples).tolist()

    return fake_images, real_images


def plot_images(state, fake_images, real_images):
    """
    Plot sampled images based on the specified state.

    Args:
    - state (str): The state for which to plot images. Should be 'fake' or 'real'.
    - fake_images (list): List of paths to three sampled fake images.
    - real_images (list): List of paths to three sampled real images.

    Raises:
    - ValueError: If the state is not 'fake' or 'real'.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    # Create subplots for three images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 10))

    # Plot images based on the specified state
    for x, ax in zip(range(3), [ax1, ax2, ax3]):
        if state == 'fake':
            image = Image.open(f'../real_vs_fake/real-vs-fake/{fake_images[x]}')
            ax.imshow(image)
            ax.set_title('Fake')
        elif state == 'real':
            image = Image.open(f'../real_vs_fake/real-vs-fake/{real_images[x]}')
            ax.imshow(image)
            ax.set_title('Real')
        else:
            raise ValueError("Enter 'real' or 'fake' as the parameter")

    # Show the plot
    plt.show()
    
def multi_histplot(df1, df2, df3):
    """
    Plots histograms of class distribution for three DataFrames representing different sets.

    Parameters:
    - df1 (DataFrame): The first DataFrame representing the class distribution of the training set.
    - df2 (DataFrame): The second DataFrame representing the class distribution of the validation set.
    - df3 (DataFrame): The third DataFrame representing the class distribution of the test set.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    fig.subplots_adjust(hspace=0.30, wspace=0.5)
    fig.suptitle('Class Distribution By Set')

    # List of titles for each subplot
    title_list = ['train_set', 'valid_set', 'test_set']

    # List of indices for each subplot
    axis_list = [(0, 0), (0, 1), (0, 2)]

    # List of DataFrames for each set
    df_list = [df1, df2, df3]

    # Iterate over sets and plot histograms
    for x in range(3):
        sns.histplot(data=df_list[x], x='label_str', ax=axes[x]).set(title=title_list[x])

def multi_scatter():
    """
    Create a multi-scatter plot showing the relationship between image size and set.

    This function uses matplotlib and seaborn to generate subplots for each set, displaying scatter plots
    of image size distribution. The x_values and y_values represent image width and height, respectively,
    for the train, valid, and test sets.

    Returns:
    - None
    """
    # Import necessary libraries
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create subplots for a multi-scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    fig.subplots_adjust(hspace=0.30, wspace=0.5)
    fig.suptitle('Image Size By Set')

    # Define title and axis information
    title_list = ['train_set', 'valid_set', 'test_set']
    x_values = [train_x, valid_x, test_x]
    y_values = [train_y, valid_y, test_y]

    # Iterate through the sets and create scatter plots
    for x in range(3):
        sns.scatterplot(x=x_values[x], y=y_values[x], ax=axes[x]).set(title=title_list[x])

def color_hist(image):
    """
    Generate a histogram for each color channel in the input image.

    This function uses NumPy and Matplotlib to create separate histograms for each color channel
    (Red, Green, and Blue) in the input image. The histograms illustrate the distribution of pixel
    intensities for each color channel.

    Parameters:
    - image (PIL.Image.Image): The input image for which histograms will be generated.

    Returns:
    - None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert the PIL Image to a NumPy array
    array = np.array(image)

    # Create histograms for each color channel
    for x in range(3):
        plt.hist(array[:, :, x].ravel(), bins=256, color=f'C{x}', alpha=0.6)

    # Display the histograms
    plt.show()
