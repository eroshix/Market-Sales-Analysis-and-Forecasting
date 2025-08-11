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
    # 1) Ham veriyi yükle
    df_raw = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    print("[main] Veri yükleme tamam.")

    # 2) Veriyi temizle
    df_clean = clean_data(df_raw)
    print("[main] Veri temizleme tamam.")

    # 3) Görselleştirmeler
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
    print("[main] Görselleştirmeler tamam.")


    # 4) Modeling 
    print("[main] Modeling başlıyor...")
    run_modeling(df_clean)
    print("[main] Modeling tamam.")

    # 5) Zaman Serisi (basit hareketli ortalama tahmini)
    print("[main] Time Series başlıyor...")
    run_time_series(df_clean)
    print("[main] Time Series tamam.")


if __name__ == "__main__":
    main()