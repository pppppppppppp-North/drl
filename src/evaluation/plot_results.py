from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_equity_curves(run_dir: Path) -> pd.DataFrame:
    paths = sorted(run_dir.glob("*_equity_curve.csv"))
    baseline_path = run_dir / "baselines" / "baseline_equity_curves.csv"
    if baseline_path.exists():
        paths.append(baseline_path)
    if not paths:
        raise FileNotFoundError(f"missing equity curve files under {run_dir}")

    curves = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    curves["date"] = pd.to_datetime(curves["date"])
    return curves.sort_values(["policy", "date"]).reset_index(drop=True)


def _load_actions(run_dir: Path) -> pd.DataFrame | None:
    paths = sorted(run_dir.glob("*_actions.csv"))
    if not paths:
        return None
    actions = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    actions["date"] = pd.to_datetime(actions["date"])
    return actions.sort_values(["policy", "ticker", "date"]).reset_index(drop=True)


def _with_drawdown(curves: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for policy, part in curves.groupby("policy", sort=False):
        part = part.copy()
        running_peak = part["portfolio_value"].cummax()
        part["plot_drawdown"] = part["portfolio_value"] / running_peak - 1.0
        part["policy"] = policy
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def plot_equity_curves(curves: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, part in curves.groupby("policy", sort=False):
        ax.plot(part["date"], part["portfolio_value"], label=policy, linewidth=1.8)
    ax.set_title("Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_drawdowns(curves: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawdowns = _with_drawdown(curves)
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, part in drawdowns.groupby("policy", sort=False):
        ax.plot(part["date"], part["plot_drawdown"], label=policy, linewidth=1.6)
    ax.set_title("Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_turnover(curves: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for policy, part in curves.groupby("policy", sort=False):
        if "turnover" not in part:
            continue
        ax.plot(part["date"], part["turnover"], label=policy, linewidth=1.4)
    ax.set_title("Portfolio Turnover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_action_mean_weights(actions: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = (
        actions.groupby(["policy", "ticker"], as_index=False)["weight"]
        .mean()
        .pivot(index="ticker", columns="policy", values="weight")
        .fillna(0.0)
        .sort_index()
    )
    ax = summary.plot(kind="bar", figsize=(10, 6), width=0.82)
    ax.set_title("Mean Action Weights")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Mean weight")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def write_result_plots(run_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = _load_equity_curves(run_dir)
    outputs = [
        output_dir / "equity_curves.png",
        output_dir / "drawdowns.png",
        output_dir / "turnover.png",
    ]
    plot_equity_curves(curves, outputs[0])
    plot_drawdowns(curves, outputs[1])
    plot_turnover(curves, outputs[2])

    actions = _load_actions(run_dir)
    if actions is not None:
        action_path = output_dir / "action_mean_weights.png"
        plot_action_mean_weights(actions, action_path)
        outputs.append(action_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    outputs = write_result_plots(run_dir, output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
