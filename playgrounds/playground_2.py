import torch
import clip
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/16", device=device)
print(model)


def interpolate_pos_encoding(positional_embedding, w, h, patch_size):
    N, dim = positional_embedding.shape
    N -= 1  # remove class emb
    class_positional_embedding = positional_embedding[0][None]
    patch_positional_embedding = positional_embedding[
        1:,
    ]

    w0 = w // patch_size
    h0 = h // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    print(math.sqrt(N))
    patch_positional_embedding = F.interpolate(
        patch_positional_embedding.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )
    assert (
        int(w0) == patch_positional_embedding.shape[-2]
        and int(h0) == patch_positional_embedding.shape[-1]
    )
    patch_positional_embedding = patch_positional_embedding.permute(0, 2, 3, 1).view(
        1, -1, dim
    )
    return torch.cat(
        (class_positional_embedding.unsqueeze(0), patch_positional_embedding), dim=1
    )


with torch.no_grad():
    print(model.visual.positional_embedding.shape)
    print(
        interpolate_pos_encoding(
            model.visual.positional_embedding, 640, 640, patch_size=16
        ).shape
    )
    # model.positional_embedding = F.interpolate(model.positional_embedding, size=(1601))

    f = model.encode_image(torch.randn((1, 3, 224, 224), device=device)).half()
    print(f.shape)
