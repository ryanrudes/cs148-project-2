from torchvision import transforms as T

COLOR = True
SIZE = 28

if COLOR:
    preprocess = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((SIZE, SIZE)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
else:
    preprocess = T.Compose([
        T.Grayscale(),
        T.Resize((SIZE, SIZE)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

augment = T.Compose([
    T.RandomRotation(10),
])
