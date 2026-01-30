import pandas as pd
import numpy as np

def generate_combined_comparison_latex(csv_file_path, output_filename):
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return

    panels = [
        {
            "type": "inflation",
            "title": "Homogeneous Variance Inflation",
            "params": [1.1, 1.2, 1.3, 1.4, 1.5]
        },
        {
            "type": "ar1",
            "title": "Correlation Structure Change",
            "params": [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        {
            "type": "spike",
            "title": "Heterogeneous Variance Inflation",
            "params": [2.0, 2.5, 3.0, 3.5, 4.0]
        }
    ]

    symbol_map = {
        'inflation': r'$\sigma^2$',
        'ar1': r'$\rho$',
        'spike': r'$\delta$'
    }

    methods = ["Our", "A19", "LL23"]

    def get_cell_content(sub_df):
        if sub_df.empty:
            return "-"
        positive_edd = sub_df[sub_df['EDD'] > 0]['EDD']
        n_positive = len(positive_edd)
        power = n_positive / 500.0
        if n_positive == 0:
            mean_val = 0.0
        else:
            mean_val = positive_edd.mean()
        return f"{mean_val:.2f} ({power:.3f})"

    print(f"Generating combined table to '{output_filename}'...")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        
        f.write(r"\begin{tabular}{lccccc}" + "\n")
        f.write(r"\toprule" + "\n")

        for i, panel in enumerate(panels):
            c_type = panel['type']
            title = panel['title']
            params = panel['params']
            symbol = symbol_map.get(c_type, '')
            # 1. Panel Title
            f.write(rf"\multicolumn{{6}}{{c}}{{{title} (Change Magnitude {symbol})}} \\" + "\n")
            f.write(r"\cmidrule(lr){2-6}" + "\n")

            # 2. Panel Header Row
            header_strs = []
            for p in params:
                if float(p).is_integer():
                    header_strs.append(str(int(p)))
                else:
                    header_strs.append(str(p))
            
            f.write(r"Method & " + " & ".join(header_strs) + r" \\" + "\n")
            f.write(r"\midrule" + "\n")

            # 3. Panel Data Rows
            for method in methods:
                row_cells = [rf"\textsf{{{method}}}"]
                for param in params:
                    mask = (
                        (df['ChangeType'] == c_type) &
                        (df['method'] == method) &
                        (np.isclose(df['ChangeParam'], param))
                    )
                    sub_df = df[mask]
                    row_cells.append(get_cell_content(sub_df))
                
                f.write(" & ".join(row_cells) + r" \\" + "\n")

            # Separator logic: if not the last panel, add middle line; if the last panel, add bottom line
            if i < len(panels) - 1:
                f.write(r"\midrule" + "\n")
            else:
                f.write(r"\bottomrule" + "\n")

        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Comparison of EDD and Power (in parentheses) with existing methods (\textsf{A19}: \citet{Avanesov2019structural}, \textsf{LL23}: \citet{Li2023online}).}" + "\n")
        f.write(r"\label{tab:comparison}" + "\n")
        f.write(r"\end{table}" + "\n")

    print("Done.")

if __name__ == "__main__":
    csv_path = '../data/compare_results.csv'
    generate_combined_comparison_latex(csv_path,output_filename="table_comparison.tex")
