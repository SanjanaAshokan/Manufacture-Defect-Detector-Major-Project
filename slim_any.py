import sys, torch
arch = sys.argv[1]
path = f"outputs/checkpoints/best_{arch}.pth"
ck = torch.load(path, map_location="cpu", weights_only=False)
sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model_state_dict"].items()}
torch.save({"model_state_dict": sd, "epoch": ck.get("epoch", 0), "metrics": ck.get("metrics", {})}, path)
print(arch, "saved")