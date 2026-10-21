import argparse
import sys
from pathlib import Path

from agents.eda_agent.artifact_tools import (
    get_run_directory,
    load_eda_report,
    save_eda_report,
)
from agents.eda_agent.eda_tools import summarize_csv
from agents.eda_agent.plotting_tools import (
    save_categorical_bar_chart,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check EDA processing and saved artifacts."
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV used for this check.",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV text encoding.",
    )

    parser.add_argument(
        "--sep",
        default=",",
        help="Character separating values.",
    )

    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Positive integer identifying this test run.",
    )

    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("workspace"),
        help="Parent folder for run artifacts.",
    )

    parser.add_argument(
    "--plot-column",
    help="Categorical column to visualize.",
    )
    
    args = parser.parse_args()

    try:
        run_dir = get_run_directory(
            args.run_id,
            args.artifact_root,
        )

        if (run_dir / "eda_report.json").exists():
            raise FileExistsError(
                "An EDA report already exists for this run."
            )

        with args.csv_path.open("rb") as csv_file:
            result = summarize_csv(
                csv_file,
                encoding=args.encoding,
                sep=args.sep,
            )

        if args.plot_column is not None:
            plot = save_categorical_bar_chart(
                result=result,
                column_name=args.plot_column,
                run_dir=run_dir,
            )

            result.plots.append(plot)

        report_path = save_eda_report(
            result=result,
            run_id=args.run_id,
            artifact_root=args.artifact_root,
        )

        loaded_result = load_eda_report(report_path)

        if loaded_result != result:
            raise ValueError(
                "The saved report differs from the calculated result."
            )

    except FileExistsError:
        print(
            "An EDA report already exists for this run. "
            "Use a new run ID.",
            file=sys.stderr,
        )
        return 1

    except (OSError, ValueError, LookupError) as error:
        print(
            f"EDA check failed: {error}",
            file=sys.stderr,
        )
        return 1

    print(result.model_dump_json(indent=2))

    print(
        f"\nSaved and reloaded successfully: {report_path}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())