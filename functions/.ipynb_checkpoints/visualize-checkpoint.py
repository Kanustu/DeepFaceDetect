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
    meta = pd.read_csv(f'./Metadata/{state}.csv')

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
            image = Image.open(f'./real_vs_fake/real-vs-fake/{fake_images[x]}')
            ax.imshow(image)
            ax.set_title('Fake')
        elif state == 'real':
            image = Image.open(f'./real_vs_fake/real-vs-fake/{real_images[x]}')
            ax.imshow(image)
            ax.set_title('Real')
        else:
            raise ValueError("Enter 'real' or 'fake' as the parameter")

    # Show the plot
    plt.show()
    
def multi_hist_plot(df, column, axis, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    """
    Display a histogram plot for a specified column within the DataFrame on a specified axis.

    Parameters:
    - df: DataFrame containing the dataset.
    - column: Name of the column in the DataFrame 'df' to be plotted.
    - axis: Axis on which to plot the histogram.

    Returns:
    - None
    """
    sns.histplot(df, x=column, ax=axes[axis]).set(title=title)

    