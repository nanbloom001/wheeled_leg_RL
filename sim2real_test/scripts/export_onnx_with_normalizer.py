"""
导出 WAVEGO actor 网络为 ONNX + normalizer 统计量为 npz。

用法 (在开发机上)：
    cd /home/user/IsaacLab
    python sim2real_test/scripts/export_onnx_with_normalizer.py \
        --checkpoint logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt \
        --output-dir sim2real_test/config/

输出：
    wavego_policy.onnx      — 纯 actor MLP (输入 48 → 输出 12)
    normalizer_stats.npz    — obs_mean(48), obs_std(48)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


NUM_OBS = 48
NUM_ACTIONS = 12


class ActorMLP(nn.Module):
    """与 rsl_rl ActorCritic 的 actor 结构完全一致。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_OBS, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main():
    parser = argparse.ArgumentParser(description="Export WAVEGO policy to ONNX + normalizer NPZ")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/rsl_rl/wavego_flat/2026-02-12_03-17-29/model_2999.pt",
        help="Path to training checkpoint (model_XXXX.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sim2real_test/config/",
        help="Output directory for onnx and npz files",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version (11 for max compatibility with Jetson)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载 checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # --- 提取 actor 权重 ---
    actor = ActorMLP()
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            actor_state[key.replace("actor.", "net.")] = value
    actor.load_state_dict(actor_state)
    actor.eval()

    # --- 验证 actor 推理 ---
    dummy_input = torch.zeros(1, NUM_OBS)
    with torch.no_grad():
        test_output = actor(dummy_input)
    print(f"Actor test output shape: {test_output.shape}  (expected [1, {NUM_ACTIONS}])")
    assert test_output.shape == (1, NUM_ACTIONS), "Actor output shape mismatch!"

    # --- 导出 ONNX ---
    onnx_path = output_dir / "wavego_policy.onnx"
    torch.onnx.export(
        actor,
        dummy_input,
        str(onnx_path),
        opset_version=args.opset_version,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes=None,  # 固定 batch_size=1
    )
    print(f"ONNX exported: {onnx_path} (opset={args.opset_version})")

    # --- 提取 normalizer 统计量 ---
    mean = state_dict["actor_obs_normalizer._mean"].cpu().numpy().astype(np.float32).reshape(-1)
    std = state_dict["actor_obs_normalizer._std"].cpu().numpy().astype(np.float32).reshape(-1)
    count = int(state_dict["actor_obs_normalizer.count"].item())

    assert mean.shape == (NUM_OBS,), f"Mean shape mismatch: {mean.shape}"
    assert std.shape == (NUM_OBS,), f"Std shape mismatch: {std.shape}"

    npz_path = output_dir / "normalizer_stats.npz"
    np.savez(
        npz_path,
        mean=mean,
        std=std,
        count=np.array([count]),
    )
    print(f"Normalizer saved: {npz_path}")
    print(f"  mean[:4] = {mean[:4]}")
    print(f"  std[:4]  = {std[:4]}")
    print(f"  count    = {count}")

    # --- 验证 ONNX 与 PyTorch 一致性 ---
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path))
        inp_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        test_obs = np.random.randn(1, NUM_OBS).astype(np.float32)

        # PyTorch
        with torch.no_grad():
            pt_out = actor(torch.from_numpy(test_obs)).numpy()

        # ONNX
        onnx_out = sess.run([out_name], {inp_name: test_obs})[0]

        max_diff = np.abs(pt_out - onnx_out).max()
        print(f"PyTorch vs ONNX max diff: {max_diff:.2e} (should be < 1e-5)")
        assert max_diff < 1e-4, f"Export mismatch too large: {max_diff}"
        print("ONNX consistency check PASSED")
    except ImportError:
        print("onnxruntime not available, skipping consistency check")

    print("\n=== Export complete ===")
    print(f"Files:")
    print(f"  {onnx_path}")
    print(f"  {npz_path}")
    print(f"\nTransfer to Jetson Nano:")
    print(f"  scp -r sim2real_test/ jetson@<IP>:~/wavego/")


if __name__ == "__main__":
    main()
