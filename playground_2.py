import torch
import clip
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/16", device=device)


with torch.no_grad():
    # model.positional_embedding = F.interpolate(model.positional_embedding, size=(1601))

    f = model.encode_image(torch.randn((1, 3, 224, 224), device=device)).half()
    print(f.shape)
