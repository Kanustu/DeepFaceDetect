def image_generator(state):
    image_gen.flow_from_directory(
    base_path + f'{state}/',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)