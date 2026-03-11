import argparse
import pandas as pd


def summarize(path: str):
    df = pd.read_csv(path)
    if df.empty:
        print(f"[ERROR] Empty CSV: {path}")
        return

    print(f"\n=== Summary: {path} ===")
    print(f"steps: {len(df)}")
    print(f"resets(max): {int(df['resets'].max()) if 'resets' in df.columns else 0}")

    key_cols = [
        "vel_x", "joint_err_mean", "joint_err_max", "torque_max", "sat_ratio",
        "dact_norm", "obs_max", "clip_dims", "sim_hz", "wall_hz"
    ]
    for c in key_cols:
        if c in df.columns:
            print(f"{c:>14s}: mean={df[c].mean():.4f}, p95={df[c].quantile(0.95):.4f}, max={df[c].max():.4f}")

    if "sat_ratio" in df.columns:
        hi = (df["sat_ratio"] >= 0.8).mean() * 100
        print(f"{'sat>=80%':>14s}: {hi:.1f}% of steps")

    if "resets" in df.columns:
        reset_points = df[df["resets"].diff().fillna(0) > 0].index.tolist()
        print(f"reset_points: {reset_points}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", help="CSV files from sim2sim_mujoco.py --log-csv")
    args = parser.parse_args()
    for p in args.csv:
        summarize(p)


if __name__ == "__main__":
    main()
