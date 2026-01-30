import pandas as pd

def empirical_size_to_latex(
    csv_path,
    output_tex
):
    # ===============================
    # 1. Read CSV (column names are provided by you)
    # ===============================
    df = pd.read_csv(csv_path)

    # ===============================
    # 2. Display mapping for weights
    # ===============================
    weight_map = {
        "gamma_0.0":  r"$\rho_{1,0}$",
        "gamma_0.25": r"$\rho_{1,0.25}$",
        "gamma_0.45": r"$\rho_{1,0.45}$",
        "log":       r"$\rho_{2}$"
    }

    dist_order   = ["Gaussian", "Uniform", "TDist"]
    weight_order = ["gamma_0.0", "gamma_0.25", "gamma_0.45", "log"]
    func_order   = ["linear", "log", "mix", "square"]

    # ===============================
    # 3. Force sorting (to avoid interference by CSV order)
    # ===============================
    df["distribution"] = pd.Categorical(
        df["distribution"], dist_order, ordered=True
    )
    df["weight_function"] = pd.Categorical(
        df["weight_function"], weight_order, ordered=True
    )
    df["function_type"] = pd.Categorical(
        df["function_type"], func_order, ordered=True
    )

    df = df.sort_values(
        ["distribution", "weight_function", "function_type"]
    )

    # ===============================
    # 4. Pivot to wide table
    # ===============================
    pivot = df.pivot_table(
        index=["distribution", "weight_function"],
        columns="function_type",
        values="empirical_size",
        aggfunc="first",
        observed=False,
    ).reset_index()

    # ===============================
    # 5. Generate LaTeX
    # ===============================
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"& & \multicolumn{4}{c}{Test Functions} \\")
    lines.append(r"\cmidrule(lr){3-6}")
    lines.append(r"Distribution & Weight & linear & log & mix & square \\")
    lines.append(r"\midrule")

    for i, dist in enumerate(dist_order):
        dist_df = pivot[pivot["distribution"] == dist]

        # Distribution multirow
        lines.append(rf"\multirow{{4}}{{*}}{{{dist}}}")

        for j, w in enumerate(weight_order):
            row = dist_df[dist_df["weight_function"] == w]

            if row.empty:
                vals = ["-"] * 4
            else:
                vals = []
                for f in func_order:
                    v = row.iloc[0].get(f)
                    if pd.isna(v):
                        vals.append("-")
                    else:
                        vals.append(f"{v:.4f}")

            weight_str = weight_map[w]

            lines.append(
                f" & {weight_str} & "
                + " & ".join(vals)
                + r" \\"
            )

        if i < len(dist_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Empirical size under different test function, weighted functions and data distributions. The nominal level is $\alpha=0.05$.}"
    )
    lines.append(r"\label{tab:empirical_size}")
    lines.append(r"\end{table}")

    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to {output_tex}")

if __name__ == "__main__":
    empirical_size_to_latex(
        csv_path="../data/H0_results.csv",
        output_tex="table_H0_results.tex"
    )
