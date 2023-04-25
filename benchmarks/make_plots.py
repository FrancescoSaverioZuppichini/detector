import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re

plt.style.use(["science"])


def main():
    parser = argparse.ArgumentParser(
        description="Create a bar plot using a column in a CSV file with respect to another column, represented by a third column."
    )
    parser.add_argument("csv_file", help="The input CSV file")
    parser.add_argument("columnA", help="The column to represent each bar")
    parser.add_argument("x_column", help="The column for the x-axis labels")
    parser.add_argument("y_column", help="The column for the y-axis values")
    parser.add_argument("output_file", help="The output file for the bar plot")
    parser.add_argument(
        "--output_dir", default=".", help="The output directory for the bar plot"
    )
    parser.add_argument("--title", help="The title for the bar plot")
    parser.add_argument(
        "--filter", help="Filter condition for the DataFrame (e.g., \"device=='cpu'\")"
    )
    parser.add_argument(
        "--rename-from-filter",
        help="Will change the name under columnA to `columnA_<filter>` in every entry that matched the filter",
        action="store_true",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    if args.filter:
        if args.rename_from_filter:
            column_names = re.findall(r"\b\w+(?=\s*==)", args.filter)
            column_name_concat = "-".join(column_names)
            df.loc[df.eval(args.filter), args.columnA] = (
                df.loc[df.eval(args.filter), args.columnA] + "-" + column_name_concat
            )
        else:
            df = df.loc[df.eval(args.filter)]

    plot = df.pivot_table(
        index=args.x_column, columns=args.columnA, values=args.y_column
    ).plot.bar(rot=0, figsize=(10, 6))

    plt.xlabel(args.x_column)
    plt.ylabel(args.y_column)

    if args.title:
        plt.title(args.title)
    else:
        plt.suptitle(
            f"{args.y_column} for each {args.columnA} by {args.x_column}", fontsize=16
        )
        filter_text = (
            " (Filtered by: {})".format(args.filter.replace("&", "\\&"))
            if args.filter
            else ""
        )
        plt.title(filter_text, fontsize=12)

    plt.savefig(f"{args.output_dir}/{args.output_file}", dpi=800)


if __name__ == "__main__":
    main()
