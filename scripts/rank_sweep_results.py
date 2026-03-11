import argparse
import glob
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="tmp/sweep/*.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.glob)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        name = os.path.basename(path).replace(".csv", "")
        rows.append(
            {
                "name": name,
                "steps": len(df),
                "resets": int(df["resets"].max()) if "resets" in df.columns else 0,
                "sat_mean": float(df["sat_ratio"].mean()),
                "sat_p95": float(df["sat_ratio"].quantile(0.95)),
                "qerr_mean": float(df["joint_err_mean"].mean()),
                "qerr_p95": float(df["joint_err_mean"].quantile(0.95)),
                "vel_abs_mean": float(df["vel_x"].abs().mean()),
                "clip_dims_mean": float(df["clip_dims"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No CSV found")
        return

    out["score"] = (
        5.0 * out["resets"]
        + 4.0 * out["sat_mean"]
        + 2.0 * out["qerr_mean"]
        + 0.5 * out["vel_abs_mean"]
        + 0.2 * out["clip_dims_mean"]
    )
    out = out.sort_values("score")

    print("=== Top 12 by score (lower better) ===")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
