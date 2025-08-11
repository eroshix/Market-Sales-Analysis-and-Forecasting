import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_distribution(df: pd.DataFrame, value_col: str = "Sales", bins: int = 30):
    plt.figure(figsize=(9, 5))
    plt.hist(df[value_col], bins=bins, color='pink', edgecolor='black')
    plt.title(f"{value_col} Distribution")
    plt.xlabel(value_col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_daily_sales_trend(df: pd.DataFrame, date_col: str = "Order Date", value_col: str = "Sales"):
    daily_sales = df["Sales"].resample("D").sum()
    plt.figure(figsize=(14, 4))
    plt.plot(daily_sales, color="skyblue")
    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_weekly_sales_trend(df: pd.DataFrame, date_col: str = "Order Date", value_col: str = "Sales"):
    weekly_sales = df["Sales"].resample("W").sum()
    plt.figure(figsize=(14, 4))
    plt.plot(weekly_sales, color="orange")
    plt.title("Weekly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_monthly_sales_trend(df: pd.DataFrame, date_col: str = "Order Date", value_col: str = "Sales"):
    monthly_sales = df["Sales"].resample("ME").sum()
    plt.figure(figsize=(14, 4))
    plt.plot(monthly_sales, color="green")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_monthly_sales_by_subcategory(df: pd.DataFrame, subcat_col: str = "Sub-Category", value_col: str = "Sales"):
    monthly_subcat = (df.groupby([pd.Grouper(freq="ME"), subcat_col], observed=False)[value_col].sum().reset_index())
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=monthly_subcat, x="Order Date", y=value_col, hue=subcat_col)
    plt.title("Monthly Sales by Sub-Category")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=25)
    plt.grid(True)
    plt.legend(title=subcat_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_sales_distribution_with_kde(df: pd.DataFrame, value_col: str = "Sales", bins: int = 50):
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[value_col], bins=bins, color="midnightblue")
    plt.title(f"{value_col} Histogram Distribution")
    plt.xlabel(value_col)
    plt.ylabel("Frequency")
    plt.grid(True)
    
    # KDE
    plt.subplot(1, 2, 2)
    sns.kdeplot(df[value_col], color="darkorange", fill=True)
    plt.title(f"{value_col} KDE Curve")
    plt.xlabel(value_col)
    plt.ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_top_n_cities_sales(df: pd.DataFrame, city_col: str = "City", value_col: str = "Sales", top_n: int = 10):
    city_sales = (df.groupby(city_col, observed=False)[value_col].sum().sort_values(ascending=False).head(top_n).reset_index())
    plt.figure(figsize=(12, 6))
    sns.barplot( data=city_sales, x=value_col, y=city_col, hue = city_col, palette="viridis", dodge=False)
    plt.title(f"Top {top_n} Cities by Total Sales")
    plt.xlabel("Total Sales")
    plt.ylabel("City")
    plt.tight_layout()
    plt.show()


def plot_sales_by_subcategory(df: pd.DataFrame, subcat_col: str = "Sub-Category", value_col: str = "Sales"):
    product_sales = (df.groupby(subcat_col, observed=False)[value_col].sum().sort_values(ascending=False).reset_index())
    plt.figure(figsize=(12, 6))
    sns.barplot(data=product_sales, x=value_col, y=subcat_col, palette="coolwarm")
    plt.title("Total Sales by Sub-Category")
    plt.xlabel("Total Sales")
    plt.ylabel(subcat_col)
    plt.tight_layout()
    plt.show()


def plot_total_sales_by_segment(df: pd.DataFrame, segment_col: str = "Segment", value_col: str = "Sales", palette: str = "Set2"):
    segment_sales = (df.groupby(segment_col, observed=False)[value_col].sum().reset_index())
    plt.figure(figsize=(8, 5))
    sns.barplot(data=segment_sales, x=segment_col, y=value_col, hue = segment_col, palette=palette)
    plt.title("Total Sales by Segment")
    plt.xlabel(segment_col)
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()


def plot_total_sales_by_region(df: pd.DataFrame, region_col: str = "Region", value_col: str = "Sales", palette: str = "magma"):
    region_sales = (df.groupby(region_col, observed=False)[value_col].sum().reset_index().sort_values(by = value_col, ascending = False))
    order = region_sales[region_col].tolist()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=region_sales, x=region_col, y=value_col, palette=palette, order=order)
    plt.title("Total Sales by Region")
    plt.xlabel(region_col)
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()


def plot_total_sales_by_shipping_mode(df: pd.DataFrame, ship_col: str = "Ship Mode", value_col: str = "Sales", palette: str = "Pastel1"):
    ship_sales = (df.groupby(ship_col, observed=False)[value_col].sum().reset_index())
    plt.figure(figsize=(8, 5))
    sns.barplot(data=ship_sales, x=ship_col, y=value_col, hue = "Ship Mode", palette=palette)
    plt.title("Total Sales by Shipping Mode")
    plt.xlabel(ship_col)
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()


def plot_sales_distribution_by_segment_violin(df: pd.DataFrame, segment_col: str = "Segment", value_col: str = "Sales", palette: str = "muted"):
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x=segment_col, y=value_col, hue = segment_col, palette=palette)
    plt.title("Sales Distribution by Segment (Violin Plot)")
    plt.xlabel(segment_col)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.show()


def plot_sales_trend_with_moving_average(df: pd.DataFrame, value_col: str = "Sales", window: int = 7):
    daily_sales = df[value_col].resample("D").sum()
    moving_avg = daily_sales.rolling(window=window).mean()
    plt.figure(figsize=(14, 6))
    plt.plot(daily_sales, label="Daily Sales", alpha=0.5)
    plt.plot(moving_avg, label=f"{window}-Day Moving Average", linewidth=2.5, color='orange')
    plt.title("Sales Trend with Daily Moving Average")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_total_sales_by_category(df: pd.DataFrame, category_col: str = "Category", value_col: str = "Sales", palette: str = "cubehelix"):
    product_sales_cat = (df.groupby(category_col, observed=False)[value_col].sum().reset_index())
    plt.figure(figsize=(12, 6))
    sns.barplot(data=product_sales_cat, x=value_col, y=category_col, palette=palette)
    plt.title("Total Sales by Category")
    plt.xlabel("Total Sales")
    plt.ylabel(category_col)
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaning import clean_data

    df_raw   = load_raw_data("C:\\Users\\stajyer\\Desktop\\GPT\\train.csv")
    df_clean = clean_data(df_raw)

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