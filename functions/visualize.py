def sample_images(state):
    import pandas as pd
    meta = pd.read_csv(f'./Metadata/{state}.csv')
    fake = meta[meta['label_str'] == 'fake']['path']
    real = meta[meta['label_str'] == 'real']['path']
    fake_images, real_images = [],[]
    for x,y in zip(fake.sample(3), real.sample(3)):
        fake_images.append(x)
        real_images.append(y)
    return fake_images, real_images

def plot_images(state, fake_images, real_images):
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12, 10))
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
            raise Valuerror("enter 'real' or 'fake' as the parameter")
    plt.show()
    