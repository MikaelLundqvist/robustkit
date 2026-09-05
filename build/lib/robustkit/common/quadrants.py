"""
Generic four-quadrant classification for any two numeric metrics.

Used by robustkit.information (mutual information vs. information
efficiency) and robustkit.benchmark (model stability vs. Cook-impact --
the "Robustness Map"), so every module that classifies things into
quadrants shares exactly the same thresholding logic. A table and its
corresponding plot can never disagree about which quadrant something
falls into, regardless of which pair of metrics is being classified.
"""


def classify_quadrants(df, x_col, y_col, x_threshold="median", y_threshold="median", labels=None):
    """
    Classify each row of df into one of four quadrants based on
    whether x_col and y_col are above or below a threshold.

    x_threshold / y_threshold: "median" (default) or a quantile in
        (0, 1) used as the cutoff for "high" on each axis.

    labels: optional dict with keys "high_high", "high_x_only",
        "high_y_only", "low_low" mapping to custom label strings.
        Unspecified keys fall back to those generic names.

    Returns a copy of df with a new "quadrant" column. The thresholds
    and column names used are stored in the result's `.attrs` for
    inspection or reuse (e.g. by a matching visualization function).
    """
    df = df.copy()
    default_labels = {
        "high_high": "high_high",
        "high_x_only": "high_x_only",
        "high_y_only": "high_y_only",
        "low_low": "low_low",
    }
    labels = {**default_labels, **(labels or {})}

    x_cut = df[x_col].median() if x_threshold == "median" else df[x_col].quantile(x_threshold)
    y_cut = df[y_col].median() if y_threshold == "median" else df[y_col].quantile(y_threshold)

    def _label(row):
        high_x = row[x_col] >= x_cut
        high_y = row[y_col] >= y_cut
        if high_x and high_y:
            return labels["high_high"]
        if high_y:
            return labels["high_y_only"]
        if high_x:
            return labels["high_x_only"]
        return labels["low_low"]

    df["quadrant"] = df.apply(_label, axis=1)
    df.attrs["x_col"] = x_col
    df.attrs["y_col"] = y_col
    df.attrs["x_threshold"] = x_cut
    df.attrs["y_threshold"] = y_cut
    return df
