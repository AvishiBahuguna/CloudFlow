import argparse
import sys
from pathlib import Path

from agents.eda_agent.artifact_tools import (
    get_run_directory,
    load_eda_report,
    save_eda_report,
)
from agents.eda_agent.chart_planning_tools import (
    build_chart_plan,
)
from agents.eda_agent.chart_rendering_tools import (
    render_chart_plan,
)
from agents.eda_agent.eda_tools import summarize_csv
from shared.eda_schemas import (
    ChartConfig,
    TargetConfig,
    TimeConfig,
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
        help="Categorical column to prioritize for plotting.",
    )
    parser.add_argument(
        "--target",
        help="Exact name of the target column.",
    )
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        help="Type of prediction task.",
    )
    parser.add_argument(
        "--date-column",
        help="Exact date column name for time analysis.",
    )
    parser.add_argument(
        "--time-frequency",
        choices=["day", "week", "month", "year"],
        help="Calendar grouping; defaults to month.",
    )
    parser.add_argument(
        "--max-charts",
        type=int,
        default=12,
        help="Maximum charts in the report.",
    )
    parser.add_argument(
        "--max-per-kind",
        type=int,
        default=3,
        help="Maximum charts of each type.",
    )

    args = parser.parse_args()

    if (args.target is None) != (args.task_type is None):
        parser.error(
            "--target and --task-type must be supplied together."
        )

    if (
        args.time_frequency is not None
        and args.date_column is None
    ):
        parser.error(
            "--time-frequency requires --date-column."
        )

    try:
        run_dir = get_run_directory(
            args.run_id,
            args.artifact_root,
        )

        if (run_dir / "eda_report.json").exists():
            raise FileExistsError(
                "An EDA report already exists for this run."
            )

        target = (
            None
            if args.target is None
            else TargetConfig(
                target_column=args.target,
                task_type=args.task_type,
            )
        )

        time_config = (
            None
            if args.date_column is None
            else TimeConfig(
                date_column=args.date_column,
                frequency=args.time_frequency or "month",
            )
        )

        with args.csv_path.open("rb") as csv_file:
            result = summarize_csv(
                csv_file,
                encoding=args.encoding,
                sep=args.sep,
                target=target,
                time_config=time_config,
            )

        result.chart_plan = build_chart_plan(
            result=result,
            config=ChartConfig(
                max_charts=args.max_charts,
                max_per_kind=args.max_per_kind,
            ),
            requested_column=args.plot_column,
        )

        render_chart_plan(
            result=result,
            run_dir=run_dir,
        )

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
            "An artifact already exists for this run. "
            "Use an unused run ID.",
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