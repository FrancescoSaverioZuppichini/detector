import torch

from src.tasks.loss import OneNetLoss
from src.tasks.object_detection.matcher import MinCostMatcher

criterion = OneNetLoss(2, MinCostMatcher())
torch.manual_seed(0)
num_queries = 3
num_classes = 2

indices = [(torch.tensor([0, 1]), torch.tensor([1, 0], dtype=torch.int64))]
class_logits = torch.tensor(
    [[[1.5410, -0.2934], [-2.1788, 0.5684], [-1.0845, -1.3986]]]
)
loss = criterion.get_class_loss(
    class_logits=class_logits,
    class_labels=torch.tensor([[1, 0]]),
    # left pred index, right label index
    indices=indices,
)
# has to be 0.8772 to match original implementation, it's correct
print(loss)
boxes_preds = torch.tensor(
    [[[10, 10, 50, 50], [20, 20, 80, 80], [10, 10, 50, 50]]]
).float()
print(boxes_preds.shape)
boxes_labels = torch.tensor([[[10, 10, 50, 50], [20, 20, 80, 80]]])
print(boxes_labels.shape)
loss = criterion.get_boxes_losses(
    boxes_preds=boxes_preds,
    boxes_labels=boxes_labels,
    # left pred index, right label index
    image_size=torch.tensor(640),
    indices=indices,
)
print(loss)
# has to be (tensor(1.8263), tensor(0.2500)), it's correct