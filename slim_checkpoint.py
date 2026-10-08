import torch, os

os.makedirs("slim", exist_ok=True)
ck = torch.load("outputs/checkpoints/best_convnext_tiny.pth", map_location="cpu", weights_only=False)
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model_state_dict"].items()}
torch.save({"model_state_dict": sd, "epoch": ck.get("epoch", 0), "metrics": ck.get("metrics", {})},
           "slim/best_convnext_tiny.pth")
print("Saved. Size (MB):", os.path.getsize("slim/best_convnext_tiny.pth") / 1e6)