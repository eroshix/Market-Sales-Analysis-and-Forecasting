from data_loader import load_raw_data
from data_cleaning import clean_data
from visualization import (
    plot_sales_distribution,
    plot_daily_sales_trend,
    plot_weekly_sales_trend,
    plot_monthly_sales_trend,
    plot_monthly_sales_by_subcategory,
    plot_sales_distribution_with_kde,
    plot_top_n_cities_sales,
    plot_sales_by_subcategory,
    plot_total_sales_by_segment,
    plot_total_sales_by_region,
    plot_total_sales_by_shipping_mode,
    plot_sales_distribution_by_segment_violin,
    plot_sales_trend_with_moving_average,
    plot_total_sales_by_category
)
from modeling import run_modeling
from time_series import run_time_series



def main():
    # 1) Load raw data
    df_raw = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    print("[main] Data loading complete.")

    # 2) Clean the data
    df_clean = clean_data(df_raw)
    print("[main] Data cleaning complete.")

    # 3) Visualizations
    plot_sales_distribution(df_clean)
    plot_daily_sales_trend(df_clean)
    plot_weekly_sales_trend(df_clean)
    plot_monthly_sales_trend(df_clean)
    plot_monthly_sales_by_subcategory(df_clean)
    plot_sales_distribution_with_kde(df_clean)
    plot_top_n_cities_sales(df_clean)
    plot_sales_by_subcategory(df_clean)
    plot_total_sales_by_segment(df_clean)
    plot_total_sales_by_region(df_clean)
    plot_total_sales_by_shipping_mode(df_clean)
    plot_sales_distribution_by_segment_violin(df_clean)    
    plot_sales_trend_with_moving_average(df_clean)
    plot_total_sales_by_category(df_clean)
    print("[main] Visualizations complete.")


    # 4) Modeling 
    print("[main] Modeling starting...")
    run_modeling(df_clean)
    print("[main] Modeling complete.")

    # 5) Time Series (simple moving average forecasting)
    print("[main] Time Series starting...")
    run_time_series(df_clean)
    print("[main] Time Series complete.")


if __name__ == "__main__":
    main()