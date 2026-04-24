import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# PLOT 1
# Global correlation between start and finish positions
# =====================================================
def plot_start_vs_finish(results_clean):

    # copy and cleaning
    df = results_clean.dropna(
        subset=["start_position", "finish_position"]
    ).copy()

    # valid positions
    df = df[
        (df["start_position"] > 0) &
        (df["finish_position"] > 0)
    ]

    # scatter plot
    plt.figure(figsize=(7, 6))
    plt.scatter(
        df["start_position"],
        df["finish_position"],
        alpha=0.15,
        s=12
    )

    # regression line
    # trend line - regression line
    # Error - vertical distance of a point from the line
    # The best fit line minimizes the sum of squared errors
    # np.polyfit returns coefficients (m, b) for y = m*x + b
    m, b = np.polyfit(
        df["start_position"],
        df["finish_position"],
        1
    )

    x = np.array([
        df["start_position"].min(),
        df["start_position"].max()
    ])

    plt.plot(x, m * x + b)

    plt.title("Global correlation between start and finish positions")
    plt.xlabel("Start position (start_position)")
    plt.ylabel("Final position (finish_position)")

    plt.show()

    # statistics
    print(f"Number of samples: {len(df)}")

    # Pearson correlation coefficient measures linear dependence between two variables
    # Values near 1 or -1 indicate strong positive/negative correlation, values near 0 indicate weak correlation
    corr = df["start_position"].corr(df["finish_position"])
    print("Pearson correlation:", corr)


# =====================================================
# PLOT 2
# Race dynamics by circuit
# Average absolute position change
# =====================================================
def plot_race_dynamics_by_circuit(results_clean):

    df = results_clean.dropna(subset=[
        "circuit_name",
        "start_position",
        "finish_position"
    ]).copy()

    # absolute position change
    df["abs_change"] = (
        df["start_position"] - df["finish_position"]
    ).abs()

    # number of samples per circuit
    n_per = df.groupby("circuit_name").size().rename("n")

    # average change
    avg_abs_change = df.groupby("circuit_name")["abs_change"].mean()

    circuit_stats = pd.concat(
        [n_per, avg_abs_change],
        axis=1
    ).reset_index()

    circuit_stats.columns = [
        "circuit_name",
        "n",
        "avg_abs_change"
    ]

    # filter circuits with enough data
    MIN_N = 200
    circuit_stats = circuit_stats[
        circuit_stats["n"] >= MIN_N
    ]

    # top circuits by number of samples
    TOP_K = 15
    top = (
        circuit_stats
        .sort_values("n", ascending=False)
        .head(TOP_K)
        .sort_values("avg_abs_change", ascending=False)
    )

    # plot
    plt.figure(figsize=(12, 6))
    plt.barh(
        top["circuit_name"],
        top["avg_abs_change"]
    )

    plt.xlabel("Average absolute position change")
    plt.ylabel("Circuit")
    plt.title("Plot 2: Race dynamics by circuit (Top 15, n ≥ 200)")
    plt.tight_layout()
    plt.show()

    print("\nTop circuits by race dynamics:\n")
    display(top)


# =====================================================
# PLOT 3
# Start position stability by circuit
# Probability that a driver finishes at the same position
# =====================================================
def plot_start_position_stability(results_clean):

    df = results_clean.dropna(subset=[
        "circuit_name",
        "start_position",
        "finish_position"
    ]).copy()

    # whether the position stayed the same
    df["same_position"] = (
        df["start_position"] == df["finish_position"]
    )

    # probability of keeping the position
    prob_same = (
        df.groupby("circuit_name")["same_position"]
        .mean()
        .reset_index()
    )

    # number of samples per circuit
    counts = (
        df.groupby("circuit_name")
        .size()
        .reset_index(name="n")
    )

    prob_same = prob_same.merge(counts, on="circuit_name")

    # filtering
    MIN_N = 150
    prob_same = prob_same[prob_same["n"] >= MIN_N]

    # plot
    plt.figure(figsize=(10, 6))
    plt.barh(
        prob_same["circuit_name"],
        prob_same["same_position"]
    )

    plt.xlabel("Probability of keeping start position")
    plt.title("Start stability by circuit (n ≥ 150)")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    print("\nStart position stability by circuit:\n")
    display(prob_same.sort_values("same_position", ascending=False))


# =====================================================
# PLOT 4
# Big drop risk for top grid positions (P1–P5)
# =====================================================
def plot_top5_big_drop_risk(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position"]
    ).copy()

    # position change (positive = gained positions)
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # only start positions 1–5
    top5 = df[df["start_position"] <= 5].copy()

    # big drop (lost >= 5 positions)
    top5["big_drop"] = top5["position_change"] <= -5

    # probability
    prob_drop = top5["big_drop"].mean()

    # grafikon
    plt.figure(figsize=(6, 4))
    plt.bar(["P1–P5"], [prob_drop])

    plt.ylabel("Probability of big drop (≥5 positions)")
    plt.title("Big drop risk for top grid positions")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print("Probability of big drop for P1–P5:", prob_drop)


# =====================================================
# PLOT 5
# Big gains from the bottom of the grid (P15–P20)
# =====================================================
def plot_bottom_grid_big_gain(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position"]
    ).copy()

    # promena pozicije
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # bottom part of the grid
    bottom_group = df[
        df["start_position"] >= 15
    ].copy()

    # big gain (>=5 positions)
    bottom_group["big_gain"] = (
        bottom_group["position_change"] >= 5
    )

    # probability
    prob_gain = bottom_group["big_gain"].mean()

    # grafikon
    plt.figure(figsize=(6, 4))
    plt.bar(["P15–P20"], [prob_gain])

    plt.ylabel("Probability of big gain (≥5 positions)")
    plt.title("Big gains from bottom of the grid")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    print("Probability of big gain for P15–P20:", prob_gain)
    print("Number of drivers in the analysis:", len(bottom_group))


# =====================================================
# PLOT 6
# Big gains (>= +5) by grid segments, by circuit
# (Top circuits by sample count)
# =====================================================
def plot_big_gain_by_grid_segment_and_circuit(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "circuit_name"]
    ).copy()

    # position change: positive = gained positions
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # grid segmentation
    def segment_grid(pos):
        if pos <= 5:
            return "P1–P5"
        elif pos <= 10:
            return "P6–P10"
        elif pos <= 15:
            return "P11–P15"
        else:
            return "P16–P20"

    order = ["P1–P5", "P6–P10", "P11–P15", "P16–P20"]

    df["grid_segment"] = df["start_position"].apply(segment_grid)

    # big gain (>= +5 positions)
    df["big_gain"] = df["position_change"] >= 5

    # filtering parameters
    MIN_N = 200
    TOP_CIRCUITS = 15

    # top circuits by sample count (with MIN_N filter)
    counts = (
        df.groupby("circuit_name")
        .size()
        .reset_index(name="n")
    )

    counts = (
        counts[counts["n"] >= MIN_N]
        .sort_values("n", ascending=False)
    )

    top_circuits = counts.head(TOP_CIRCUITS)["circuit_name"].tolist()
    df_top = df[df["circuit_name"].isin(top_circuits)].copy()

    # probabilities per (circuit, segment)
    stats = (
        df_top.groupby(["circuit_name", "grid_segment"])["big_gain"]
        .mean()
        .reset_index()
    )

    # ensure ordering of segments
    stats["grid_segment"] = pd.Categorical(
        stats["grid_segment"],
        categories=order,
        ordered=True
    )
    stats = stats.sort_values(["circuit_name", "grid_segment"])

    # plotting
    plt.figure(figsize=(12, 6))
    for seg in order:
        tmp = stats[stats["grid_segment"] == seg]
        plt.scatter(
            tmp["circuit_name"],
            tmp["big_gain"],
            s=90,
            label=seg
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Probability of big gain (≥5 positions)")
    plt.xlabel("Circuit")
    plt.title(
        f"Big gains (≥+5) by grid segment and circuit "
        f"(Top {TOP_CIRCUITS}, n ≥ {MIN_N})"
    )
    plt.ylim(0, 1)
    plt.legend(title="Starting segment")
    plt.tight_layout()
    plt.show()

    display(stats)

    return stats


# =====================================================
# PLOT 7
# Distribution of position changes pre vs post DRS
# =====================================================
def plot_position_change_distribution_drs(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "year"]
    ).copy()

    # apsolutna promena pozicije
    df["abs_change"] = (
        df["start_position"] - df["finish_position"]
    ).abs()

    # period (pre / post DRS)
    df["period"] = df["year"].apply(
        lambda y: "Pre-DRS (2000–2010)"
        if y <= 2010
        else "Post-DRS (2011–2024)"
    )

    plt.figure(figsize=(8, 5))

    for period, color in zip(
        ["Pre DRS (2000–2010)", "Posle DRS (2011–2024)"],
        ["#0553a0", "#e74c3c"]
    ):
        subset = df[df["period"] == period]

        plt.hist(
            subset["abs_change"],
            bins=15,
            alpha=0.5,
            label=period
        )

    plt.xlabel("Absolute position change")
    plt.ylabel("Frequency")
    plt.title("Distribution of position changes pre vs post DRS")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nNumber of samples per period:")
    print(df["period"].value_counts())

# =====================================================
# PLOT 8
# Position change pre vs post DRS (boxplot)
# =====================================================
def plot_position_change_boxplot_drs(results_clean):

    df_drs = results_clean.dropna(
        subset=["drs_period", "position_change"]
    ).copy()

    # stabilan redosled perioda
    order = [
        p for p in ["pre_drs", "post_drs"]
        if p in df_drs["drs_period"].unique()
    ]

    data = [
        df_drs.loc[
            df_drs["drs_period"] == p,
            "position_change"
        ].values
        for p in order
    ]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        data,
        tick_labels=order
    )

    plt.xlabel("Period")
    plt.ylabel("Position change (start - finish)")
    plt.title("Position change — pre vs post DRS (boxplot)")

    plt.tight_layout()
    plt.show()

    print("\nStatistics by period:")
    print(
        df_drs.groupby("drs_period")["position_change"]
        .describe()[["mean", "std", "min", "max"]]
    )

# =====================================================
# PLOT 9
# Small vs big gains pre vs post DRS
# =====================================================
def plot_small_vs_big_gains_drs(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "year"]
    ).copy()

    # promena pozicije
    df["position_change"] = (
        df["start_position"] - df["finish_position"]
    )

    # period
    df["period"] = df["year"].apply(
        lambda y: "Pre DRS" if y <= 2010 else "Post DRS"
    )

    # gain definitions
    df["small_gain"] = df["position_change"].between(1, 3)
    df["big_gain"] = df["position_change"] >= 5

    # probabilities by period
    stats = (
        df.groupby("period")[["small_gain", "big_gain"]]
        .mean()
        .reset_index()
    )

    # ensure stable ordering
    order = ["Pre DRS", "Posle DRS"]
    stats["period"] = pd.Categorical(
        stats["period"],
        categories=order,
        ordered=True
    )
    stats = stats.sort_values("period")

    # grouped bar plot
    x = np.arange(len(stats["period"]))
    w = 0.35

    plt.figure(figsize=(7, 4))

    plt.bar(
        x - w/2,
        stats["small_gain"],
        width=w,
        label="Small gain (+1 to +3)",
        color="#2ecc71"
    )

    plt.bar(
        x + w/2,
        stats["big_gain"],
        width=w,
        label="Big gain (≥ +5)",
        color="#e74c3c"
    )

    plt.xticks(x, stats["period"])
    plt.ylabel("Probability")
    plt.title("Gains pre vs post DRS (small vs big)")
    plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nGains probabilities:")
    display(stats)


# =====================================================
# PLOT 10
# Trend of start–finish correlation over time
# =====================================================
def plot_start_finish_correlation_trend(results_clean):

    df = results_clean.dropna(
        subset=["start_position", "finish_position", "year"]
    ).copy()

    # sigurnost: numeric tipovi
    df["start_position"] = pd.to_numeric(
        df["start_position"], errors="coerce"
    )
    df["finish_position"] = pd.to_numeric(
        df["finish_position"], errors="coerce"
    )

    df = df.dropna(
        subset=["start_position", "finish_position", "year"]
    )

    # korelacija po godinama
    year_corr = (
        df.groupby("year")[["start_position", "finish_position"]]
          .corr()
          .iloc[0::2, -1]   # start vs finish korelacija
          .reset_index()
    )

    year_corr.columns = ["year", "level_1", "correlation"]
    year_corr = year_corr[
        year_corr["level_1"] == "start_position"
    ][["year", "correlation"]]

    # plot
    plt.figure(figsize=(10, 5))

    plt.plot(
        year_corr["year"],
        year_corr["correlation"]
    )

    # vertical line — DRS introduction
    plt.axvline(
        2011,
        linestyle="--",
        label="DRS introduction (2011)"
    )

    plt.xlabel("Season")
    plt.ylabel("Start–finish correlation")
    plt.title("Trend of start–finish correlation over time")

    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nAverage correlation:")
    print(year_corr["correlation"].describe()[["mean", "min", "max"]])

