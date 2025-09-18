from pathlib import Path
import re
import html

import pandas as pd


# next two methods:
# clumsy number formatting to avoid
# the even clumsier usage of locale-based formatting
# which requires packages installed on the system level.
def float_str(number: float, precision: int = 2) -> str:
    """
    German floating point number without trailing zeros
    """
    return f"{number:.{precision}f}".rstrip("0").rstrip(".").replace(".", ",")


def int_str(number: float) -> str:
    """
    German integer with thousands separator
    """
    return f"{number:,.0f}".replace(",", ".")


def replace_html_title(file_path: Path, title: str) -> None:
    """
    Change the HTML title in-place
    """
    temp_file_path = file_path.with_suffix(file_path.suffix + ".tmp")
    escaped_title = html.escape(title).replace("\n", " - ")

    with (
        file_path.open("r", encoding="utf-8") as infile,
        temp_file_path.open("w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if "<title>" in line and "</title>" in line:
                line = re.sub(
                    r"(<title>)(.*?)(</title>)", rf"\1{escaped_title}\3", line
                )
            outfile.write(line)

    temp_file_path.replace(file_path)


def bin_seconds_of_day(seconds: pd.Series, bin_size) -> pd.DataFrame:
    """
    Bin timestamps (given as seconds of day) into intervals of `bin_size` seconds.

    Returns:
    - DataFrame with binned departure times and their counts.
    """
    max_time = 24 * 60 * 60  # seconds in a day
    bins = range(0, max_time + bin_size, bin_size)
    labels = [f"{int(b // 3600):02d}:{int((b % 3600) // 60):02d}" for b in bins[:-1]]
    times_binned = pd.cut(seconds, bins=bins, labels=labels, right=False)
    count = times_binned.value_counts().sort_index()
    return pd.DataFrame({"time_bin": count.index, "count": count.values}).set_index(
        "time_bin"
    )
