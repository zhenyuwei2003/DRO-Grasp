import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.network import create_network
from utils.device_utils import get_device_name, resolve_device, synchronize_device


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DRO-Grasp eager and torch.compile execution.")
    parser.add_argument("--device", choices=("auto", "xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--mode", choices=("eager", "compile", "both"), default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--block-computing", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--training-step", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def timed_forward(model, inputs, device, warmup, iterations):
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        synchronize_device(device)
        start = time.perf_counter()
        output = None
        for _ in range(iterations):
            output = model(*inputs)["dro"]
        synchronize_device(device)
    return output, (time.perf_counter() - start) * 1000 / iterations


def run_training_step(cfg, device, robot_pc, object_pc):
    model = create_network(cfg, mode="train").to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    target_pc = torch.randn_like(robot_pc)
    optimizer.zero_grad(set_to_none=True)
    output = model(robot_pc, object_pc, target_pc=target_pc)["dro"]
    loss = output.square().mean()
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Training smoke-test loss is not finite")
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("Training smoke test produced non-finite gradients")
    optimizer.step()
    return float(loss.detach().cpu())


def main():
    args = parse_args()
    if args.num_points < 32:
        raise ValueError("--num-points must be at least 32 for the encoder KNN operation")
    if args.block_computing and args.num_points % 4 != 0:
        raise ValueError("--num-points must be divisible by 4 with --block-computing")

    torch.manual_seed(42)
    device = resolve_device(args.device, args.device_index)
    cfg = SimpleNamespace(
        emb_dim=args.emb_dim,
        latent_dim=args.latent_dim,
        pretrain=None,
        center_pc=True,
        block_computing=args.block_computing,
    )
    model = create_network(cfg, mode="validate").to(device).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    robot_pc = torch.randn(args.batch_size, args.num_points, 3, device=device)
    object_pc = torch.randn(args.batch_size, args.num_points, 3, device=device)
    latent = torch.randn(args.batch_size, args.latent_dim, device=device)
    inputs = (robot_pc, object_pc, None, latent)
    results = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": get_device_name(device),
        "shape": [args.batch_size, args.num_points, args.num_points],
    }

    eager_output = None
    if args.mode in ("eager", "both"):
        eager_output, eager_ms = timed_forward(model, inputs, device, args.warmup, args.iterations)
        results["eager_ms"] = eager_ms

    if args.mode in ("compile", "both"):
        compiled_model = torch.compile(model, backend="inductor", fullgraph=False)
        compiled_output, compiled_ms = timed_forward(
            compiled_model, inputs, device, args.warmup, args.iterations
        )
        results["compiled_ms"] = compiled_ms
        if eager_output is None:
            with torch.no_grad():
                eager_output = model(*inputs)["dro"]
        difference = (eager_output.float() - compiled_output.float()).abs()
        results["max_abs_diff"] = float(difference.max().cpu())
        results["mean_abs_diff"] = float(difference.mean().cpu())
        torch.testing.assert_close(compiled_output, eager_output, rtol=1e-4, atol=1e-5)

    if eager_output is None or not torch.isfinite(eager_output).all():
        raise RuntimeError("Eager output is missing or non-finite")
    if tuple(eager_output.shape) != tuple(results["shape"]):
        raise RuntimeError(f"Unexpected output shape: {tuple(eager_output.shape)}")

    if args.training_step:
        results["training_loss"] = run_training_step(cfg, device, robot_pc, object_pc)

    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()