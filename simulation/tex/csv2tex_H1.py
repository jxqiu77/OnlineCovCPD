import pandas as pd
import numpy as np

def get_single_latex_table_str(df, config):
    """
    Generate the LaTeX string for a single table.
    """
    weight_func = config['weight']
    change_type = config['type']
    target_params = config['params']
    header_label = config['header_sym']
    caption_text = config['caption']

    # 1. Filter data
    sub_df = df[
        (df['WeightFunction'] == weight_func) & 
        (df['ChangeType'] == change_type)
    ].copy()

    if sub_df.empty:
        return f"% No data found for Weight={weight_func}, Type={change_type}"
    
    # Replace distribution names
    sub_df['Distribution'] = sub_df['Distribution'].replace({'TDist': r"\texttt{Student's $t(10)$}"})
    sub_df['Distribution'] = sub_df['Distribution'].replace({'Gaussian': r'\texttt{Gaussian}'})
    sub_df['Distribution'] = sub_df['Distribution'].replace({'Uniform': r'\texttt{Uniform}'})
    sub_df['FunctionType'] = sub_df['FunctionType'].replace({'linear': r'\texttt{linear}'})
    sub_df['FunctionType'] = sub_df['FunctionType'].replace({'log': r'\texttt{log}'})
    sub_df['FunctionType'] = sub_df['FunctionType'].replace({'mix': r'\texttt{mix}'})

    # 2. Format data: EDD keeps two decimals, Power remains as is (Power may be str or float)
    sub_df['ResultStr'] = sub_df.apply(
        lambda row: f"{row['EDD']:.2f} ({row['Power']})", axis=1
    )

    # 3. Pivot the data
    pivot = sub_df.pivot_table(
        index=['Distribution', 'FunctionType', 'K_Star'],
        columns='ChangeParam',
        values='ResultStr',
        aggfunc='first'
    ).reset_index()

    # Ensure column names are strings for matching (since ChangeParam may be float)
    pivot.columns = [str(c) for c in pivot.columns]

    # 4. Define sorting order
    dist_order = [r'\texttt{Gaussian}', r'\texttt{Uniform}', r"\texttt{Student's $t(10)$}"]
    func_order = [r'\texttt{linear}', r'\texttt{log}', r'\texttt{mix}']
    k_star_order = [350, 450, 550]

    # Set as categorical for sorting
    pivot['Distribution'] = pd.Categorical(pivot['Distribution'], categories=dist_order, ordered=True)
    pivot['FunctionType'] = pd.Categorical(pivot['FunctionType'], categories=func_order, ordered=True)
    pivot['K_Star'] = pd.Categorical(pivot['K_Star'], categories=k_star_order, ordered=True)
    
    pivot = pivot.sort_values(['Distribution', 'FunctionType', 'K_Star'])

    # 5. Generate LaTeX code
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{cclllll}")
    lines.append(r"\toprule")
    lines.append(rf"& & \multicolumn{{5}}{{c}}{{Change Magnitude ({header_label})}} \\")
    lines.append(r"\cmidrule(lr){3-7}")

    # Generate column headers (e.g. 10, 20, 30, ...)
    col_headers = []
    valid_cols = [] 
    
    for val in target_params:
        # Try to find a matching column in the pivot (handle float precision)
        found_col = None
        for col_name in pivot.columns:
            try:
                # Use np.isclose for tolerant float matching
                if np.isclose(float(col_name), float(val)):
                    found_col = col_name
                    break
            except:
                continue
        
        # Display format for the column header (integer if possible, else float)
        try:
            val_float = float(val)
            display_val = str(int(val_float)) if val_float.is_integer() else str(val_float)
        except:
            display_val = str(val)
            
        col_headers.append(r"\multicolumn{1}{c}{" + display_val + "}")
        valid_cols.append(found_col) # None if not found

    lines.append(r"& $k^{\star}$ & " + " & ".join(col_headers) + r" \\")
    lines.append(r"\midrule")

    # Generate table body
    for i, dist in enumerate(dist_order):
        # Distribution name row
        lines.append(rf"\multicolumn{{2}}{{c}}{{}}& \multicolumn{{5}}{{l}}{{{dist}}} \\")
        
        dist_df = pivot[pivot['Distribution'] == dist]
        
        for j, func in enumerate(func_order):
            func_df = dist_df[dist_df['FunctionType'] == func]
            
            first_row = True
            for _, row in func_df.iterrows():
                k_val = row['K_Star']
                row_cells = []
                for col in valid_cols:
                    if col and col in row and pd.notna(row[col]):
                        row_cells.append(str(row[col]))
                    else:
                        row_cells.append("-")
                
                content_str = " & ".join(row_cells)
                
                if first_row:
                    lines.append(rf"\multirow{{3}}{{*}}{{{func}}} & {k_val} & {content_str} \\")
                    first_row = False
                else:
                    lines.append(rf"& {k_val} & {content_str} \\")
            
            # Add separation rule after each function block (except for the last function of a distribution)
            if j < len(func_order) - 1:
                lines.append(r"\cmidrule(lr){2-7}")
        
        # Add midrule after each distribution block (except the last one)
        if i < len(dist_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption_text + "}")
    
    # Generate unique label
    label_str = f"tab:{change_type}_{weight_func}".replace(".", "")
    lines.append(rf"\label{{{label_str}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def generate_split_latex_tables(csv_file_path):
    """
    Read CSV and generate LaTeX tables, splitting them into two files:
    1. table_H1_results_gamma_0.tex (only gamma_0.0)
    2. table_H1_results_others.tex (gamma_0.25, gamma_0.45, log)
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return

    # Define all table configurations
    tables_config = [
        # === Gamma 0.0 ===
        {"weight": "gamma_0.0", "type": "inflation", "params": [1.1, 1.2, 1.3, 1.4, 1.5], "header_sym": r"$\sigma^2$", "caption": r"EDD and Power (in parentheses) for \emph{homogeneous variance inflation} using weight $\rho_{1,\gamma}$ ($\gamma=0$)."},
        {"weight": "gamma_0.0", "type": "ar1", "params": [0.1, 0.3, 0.5, 0.7, 0.9], "header_sym": r"$\rho$", "caption": r"EDD and Power (in parentheses) for \emph{correlation structure change} using weight $\rho_{1,\gamma}$ ($\gamma=0$)."},
        {"weight": "gamma_0.0", "type": "spike", "params": [2.0, 2.5, 3.0, 3.5, 4.0], "header_sym": r"$\delta$", "caption": r"EDD and Power (in parentheses) for \emph{heterogeneous variance inflation} using weight $\rho_{1,\gamma}$ ($\gamma=0$)."},
        
        # === Gamma 0.25 ===
        {"weight": "gamma_0.25", "type": "inflation", "params": [1.1, 1.2, 1.3, 1.4, 1.5], "header_sym": r"$\sigma^2$", "caption": r"EDD and Power (in parentheses) for \emph{homogeneous variance inflation} using weight $\rho_{1,\gamma}$ ($\gamma=0.25$)."},
        {"weight": "gamma_0.25", "type": "ar1", "params": [0.1, 0.3, 0.5, 0.7, 0.9], "header_sym": r"$\rho$", "caption": r"EDD and Power (in parentheses) for \emph{correlation structure change} using weight $\rho_{1,\gamma}$ ($\gamma=0.25$)."},
        {"weight": "gamma_0.25", "type": "spike", "params": [2.0, 2.5, 3.0, 3.5, 4.0], "header_sym": r"$\delta$", "caption": r"EDD and Power (in parentheses) for \emph{heterogeneous variance inflation} using weight $\rho_{1,\gamma}$ ($\gamma=0.25$)."},

        # === Gamma 0.45 ===
        {"weight": "gamma_0.45", "type": "inflation", "params": [1.1, 1.2, 1.3, 1.4, 1.5], "header_sym": r"$\sigma^2$", "caption": r"EDD and Power (in parentheses) for \emph{homogeneous variance inflation} using weight $\rho_{1,\gamma}$ ($\gamma=0.45$)."},
        {"weight": "gamma_0.45", "type": "ar1", "params": [0.1, 0.3, 0.5, 0.7, 0.9], "header_sym": r"$\rho$", "caption": r"EDD and Power (in parentheses) for \emph{correlation structure change} using weight $\rho_{1,\gamma}$ ($\gamma=0.45$)."},
        {"weight": "gamma_0.45", "type": "spike", "params": [2.0, 2.5, 3.0, 3.5, 4.0], "header_sym": r"$\delta$", "caption": r"EDD and Power (in parentheses) for \emph{heterogeneous variance inflation} using weight $\rho_{1,\gamma}$ ($\gamma=0.45$)."},

        # === log ===
        {"weight": "log", "type": "inflation", "params": [1.1, 1.2, 1.3, 1.4, 1.5], "header_sym": r"$\sigma^2$", "caption": r"EDD and Power (in parentheses) for \emph{homogeneous variance inflation} using weight $\rho_2$."},
        {"weight": "log", "type": "ar1", "params": [0.1, 0.3, 0.5, 0.7, 0.9], "header_sym": r"$\rho$", "caption": r"EDD and Power (in parentheses) for \emph{correlation structure change} using weight $\rho_2$."},
        {"weight": "log", "type": "spike", "params": [2.0, 2.5, 3.0, 3.5, 4.0], "header_sym": r"$\delta$", "caption": r"EDD and Power (in parentheses) for \emph{heterogeneous variance inflation} using weight $\rho_2$."},
    ]

    print(f"Processing {len(tables_config)} tables...")

    # Define output filenames
    file_gamma0 = "table_H1_results_gamma_0.tex"
    file_others = "table_H1_results_others.tex"

    # Open both files using 'with' statement
    with open(file_gamma0, "w", encoding="utf-8") as f_g0, \
         open(file_others, "w", encoding="utf-8") as f_ot:
        
        for config in tables_config:
            print(f"Generating: {config['weight']} - {config['type']}...")
            
            # Generate the table content
            table_str = get_single_latex_table_str(df, config)
            
            # Select which file to write to
            if config['weight'] == "gamma_0.0":
                target_file = f_g0
            else:
                target_file = f_ot

            # Write content
            target_file.write(f"% === Table: {config['weight']} - {config['type']} ===\n")
            target_file.write(table_str)
            target_file.write("\n\n\n")
    
    print(f"\nTables have been split and saved to:\n1. {file_gamma0}\n2. {file_others}")

if __name__ == "__main__":
    csv_path = '../data/H1_results.csv'
    generate_split_latex_tables(csv_path)
