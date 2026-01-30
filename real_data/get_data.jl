using HTTP, Gumbo, Cascadia, DataFrames, YFinance, Dates, ProgressMeter, CSV, Impute, Statistics, LinearAlgebra, Impute

function get_sp500_tickers()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = HTTP.get(url)
    html = parsehtml(String(response.body))
    
    # The first table on the Wikipedia page contains the list of S&P 500 components
    table = eachmatch(Selector("table#constituents"), html.root)[1]
    rows = eachmatch(Selector("tr"), table)[2:end] # Skip the header row
    
    tickers = String[]
    for row in rows
        # The first column is the Ticker Symbol
        ticker_cell = eachmatch(Selector("td"), row)[1]
        ticker = nodeText(ticker_cell)
        # Data cleaning: Wikipedia sometimes uses BF.B, but Yahoo needs BF-B
        ticker = replace(strip(ticker), "." => "-")
        push!(tickers, ticker)
    end
    
    println("Successfully fetched $(length(tickers)) tickers.")
    return tickers
end

function winsorize_matrix(X::Matrix; limit=5.0)
    # 1. Compute the mean and standard deviation for each column (each stock)
    mu = mean(X, dims=1)
    sigma = std(X, dims=1)
    X_clean = copy(X)
    p = size(X, 2)
    # 2. Clamp values in each column within [mean-limit*std, mean+limit*std]
    for j in 1:p
        lower_bound = mu[j] - limit * sigma[j]
        upper_bound = mu[j] + limit * sigma[j]
        X_clean[:, j] = clamp.(X[:, j], lower_bound, upper_bound)
    end

    return X_clean
end

"""
    process_sp500_data(df::DataFrame; missing_threshold=0.05)

Clean S&P 500 data and return the log returns matrix X (n x p).
- df: Wide table, the first column must be time, the remaining columns are stock codes.
- missing_threshold: Maximum allowed missing rate (default 5%). 
"""
function process_sp500_data(df::DataFrame; missing_threshold=0.05)
    println("Start data cleaning")
    println("Original dimension: $(size(df))")
    
    # 1. Sort by time
    time_col = names(df)[1]
    rename!(df, time_col => :timestamp) # Rename the time column to :timestamp
    sort!(df, :timestamp)
    
    # 2. Filter qualified stocks (Columns)
    # Remove columns with too many missing values
    total_rows = nrow(df)
    cols_to_keep = Symbol[:timestamp] 
    for col in names(df)[2:end]
        col_sym = Symbol(col)
        miss_rate = count(ismissing, df[!, col_sym]) / total_rows
        if miss_rate <= missing_threshold
            push!(cols_to_keep, col_sym)
        end
    end
    
    clean_df = select(df, cols_to_keep)
    println("Save stocks: $(length(cols_to_keep)-1) (Removed stocks with too many missing values)")
    
    # 3. Forward Fill
    # If today is missing, use yesterday's closing price.
    for col in names(clean_df)[2:end]
        v = clean_df[!, col]
        if !isempty(v)
            last_val = ismissing(v[1]) ? NaN : v[1] 
            for i in 1:length(v)
                if ismissing(v[i])
                    v[i] = last_val
                else
                    last_val = v[i]
                end
            end
        end
    end
    
    # 4. Delete remaining missing rows (Drop Rows)
    final_df = dropmissing(clean_df)
    println("Final retained trading days: $(nrow(final_df))")
    
    # 5. Extract the price matrix and convert to Float64
    dates = final_df.timestamp[2:end] 
    price_matrix = Matrix{Float64}(final_df[:, 2:end])
    
    # 6. Calculate the Log Returns
    # r_t = log(P_t) - log(P_{t-1})
    if any(price_matrix .<= 0)
        @warn "Detected price <= 0, which may cause log calculation error, please check the data source!"
    end
    log_returns = diff(log.(price_matrix), dims=1)
    log_returns[isnan.(log_returns)] .= 0.0;

    log_returns_clean = winsorize_matrix(log_returns, limit=5.0);

    tickers =  names(clean_df)[2:end]
    df_log_returns = DataFrame(log_returns_clean, tickers)
    insertcols!(df_log_returns, 1, :timestamp => dates)
    
    println("Data cleaning completed")
    println("Final matrix X dimension: $(size(log_returns_clean)) (n=$(size(log_returns_clean,1)), p=$(size(log_returns_clean,2)))")
    return df_log_returns, dates, tickers
end

sp500_tickers = get_sp500_tickers()
start_date = "2019-01-01"
end_date = "2022-01-01"
stock_data = Dict{String, DataFrame}()

# Use progress bar to batch download
@showprogress "[Downloading data]" for ticker in sp500_tickers[1:end]
    # Download daily data
    data = get_prices(ticker, startdt=start_date, enddt=end_date)
    
    # YFinance returns an OrderedDict, need to convert to DataFrame
    if !isempty(data)
        # Convert to DataFrame
        df = DataFrame(data)
        
        # YFinance returns the column names: :timestamp, :open, :high, :low, :close, :adjclose, :volume
        # We only need timestamp and adjclose
        if hasproperty(df, :timestamp) && hasproperty(df, :adjclose)
            df = select(df, :timestamp, :adjclose)
            rename!(df, :adjclose => Symbol(ticker)) # Rename to the stock code
            stock_data[ticker] = df
        else
            println("Warning: $ticker data format is not expected")
        end
    end

    sleep(0.1) # Pause for a moment to avoid triggering Yahoo's rate limit
end

dfs = values(stock_data);
full_df = reduce((x, y) -> outerjoin(x, y, on=:timestamp), dfs);
log_returns, dates, tickers = process_sp500_data(full_df);

cd(@__DIR__)
CSV.write("sp500_full.csv", full_df)
CSV.write("sp500_log_returns.csv", log_returns)
