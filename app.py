
import os
import io
import sys
import warnings

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Monkey-patch: make any matplotlib/seaborn plt.show() render inside Streamlit ---
def _st_show(*args, **kwargs):
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close()
plt.show = _st_show

# So local imports work when the app is launched from another folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Project modules
from data_loader import load_raw_data
from data_cleaning import clean_data
import visualization as viz
from modeling import run_modeling
from time_series import run_time_series


st.set_page_config(page_title="Sales Analytics — Streamlit", layout="wide")
st.title("📊 Sales Analytics & Forecasting")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    data_mode = st.radio("Veri kaynağı seç", ["Projede duran train.csv", "CSV yükle (upload)"], index=0)
    default_path = st.text_input(
        "Dosya yolu (projede duran dosya için)",
        value="train.csv",
        help="app.py ile aynı klasördeyse sadece 'train.csv' yazmanız yeterli."
    )
    uploaded = None
    if data_mode == "CSV yükle (upload)":
        uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"])

    st.markdown("---")
    st.caption("Grafik çizimler biraz zaman alabilir. Sabırlı olun 😊")

@st.cache_data(show_spinner=False)
def _read_uploaded_csv(file_like) -> pd.DataFrame:
    return pd.read_csv(file_like)

@st.cache_data(show_spinner=False)
def _load_default_csv(path: str) -> pd.DataFrame:
    # load_raw_data fonksiyonunu kullanarak okuyoruz
    return load_raw_data(path)

def _load_and_clean():
    if uploaded is not None:
        df_raw = _read_uploaded_csv(uploaded)
    else:
        df_raw = _load_default_csv(default_path)

    df_clean = clean_data(df_raw)
    return df_raw, df_clean

# --- DATA ---
st.subheader("1) Veri Yükleme & Temizleme")
if st.button("Veriyi yükle ve temizle", type="primary"):
    with st.spinner("Yükleniyor ve temizleniyor..."):
        try:
            df_raw, df_clean = _load_and_clean()
            st.success("Veri hazır!")
            st.session_state["df_raw"] = df_raw
            st.session_state["df_clean"] = df_clean
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

if "df_clean" in st.session_state:
    df_raw = st.session_state["df_raw"]
    df_clean = st.session_state["df_clean"]

    st.write("**Ham veri (ilk 5 satır):**")
    st.dataframe(df_raw.head(), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Satır sayısı (ham)", len(df_raw))
    with c2: st.metric("Sütun sayısı (ham)", df_raw.shape[1])
    with c3: st.metric("Satır sayısı (temiz)", len(df_clean))
    with c4: st.metric("Sütun sayısı (temiz)", df_clean.shape[1])

    with st.expander("Sütun tipleri (temiz veri)"):
        dtypes_df = pd.DataFrame({"column": df_clean.columns, "dtype": df_clean.dtypes.astype(str)})
        st.dataframe(dtypes_df, use_container_width=True)

    st.subheader("2) Keşifsel Görselleştirmeler")
    st.caption("İstediğin grafikleri seç ve çiz.")
    # Seçim kutuları
    viz_options = {
        "Satış dağılımı (histogram)": viz.plot_sales_distribution,
        "Günlük satış trendi": viz.plot_daily_sales_trend,
        "Haftalık satış trendi": viz.plot_weekly_sales_trend,
        "Aylık satış trendi": viz.plot_monthly_sales_trend,
        "Aylık satışlar (Alt-kategori bazında)": viz.plot_monthly_sales_by_subcategory,
        "Satış dağılımı + KDE": viz.plot_sales_distribution_with_kde,
        "En çok satış yapan Top-N şehirler": viz.plot_top_n_cities_sales,
        "Alt-kategori bazında toplam satış": viz.plot_sales_by_subcategory,
        "Segment bazında toplam satış": viz.plot_total_sales_by_segment,
        "Bölge bazında toplam satış": viz.plot_total_sales_by_region,
        "Kargo modu bazında toplam satış": viz.plot_total_sales_by_shipping_mode,
        "Segment bazında violin dağılım": viz.plot_sales_distribution_by_segment_violin,
        "Hareketli ortalama trendi": viz.plot_sales_trend_with_moving_average,
        "Kategori bazında toplam satış": viz.plot_total_sales_by_category,
    }

    selected_viz = st.multiselect("Çizilecek grafik(ler)i seç", list(viz_options.keys()))
    if st.button("Seçili grafikleri çiz"):
        with st.spinner("Grafikler çiziliyor..."):
            for k in selected_viz:
                st.markdown(f"**{k}**")
                try:
                    viz_options[k](df_clean)
                except Exception as e:
                    st.error(f"{k} çizilirken hata: {e}")

    st.subheader("3) Basit Modelleme (Linear Regression)")
    st.caption("Mevcut kodlarda iki farklı lineer model deneniyor ve metrikler yazdırılıyor.")
    if st.button("Modellemeyi çalıştır"):
        with st.spinner("Eğitiliyor..."):
            try:
                out = run_modeling(df_clean)
                if isinstance(out, dict) and "metrics_last" in out:
                    m = out["metrics_last"]
                    st.success("Modelleme tamamlandı!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("MSE", f'{m.get("mse", float("nan")):.3f}')
                    c2.metric("RMSE", f'{m.get("rmse", float("nan")):.3f}')
                    c3.metric("R²", f'{m.get("r2", float("nan")):.3f}')
                    c4.metric("MAPE (%)", f'{m.get("mape", float("nan")):.2f}')
                else:
                    st.info("Fonksiyon metrik sözlüğü döndürmedi; konsol çıktısına bakın.")
            except Exception as e:
                st.error(f"Modelleme çalışırken hata: {e}")

    st.subheader("4) Zaman Serisi — 7 Gün Tahmin (Prophet & SARIMA)")
    st.caption("Hazır fonksiyon 7 günlük test tahmini yapar ve karşılaştırır.")
    if st.button("Zaman serisi analizi & tahminleri çalıştır"):
        with st.spinner("Prophet/SARIMA eğitim ve tahmin yapılıyor..."):
            try:
                ts_out = run_time_series(df_clean)
                if isinstance(ts_out, dict):
                    p = ts_out.get("prophet", {})
                    s = ts_out.get("sarima", {})
                    st.success("Tahmin tamamlandı!")
                    st.markdown("**Özet metrikler (Test Seti)**")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Prophet RMSE", f'{p.get("rmse", float("nan")):.2f}')
                    cc2.metric("Prophet MAPE (%)", f'{p.get("mape", float("nan")):.2f}')
                    cc3.metric("Prophet R²", f'{p.get("r2", float("nan")):.2f}')

                    dd1, dd2, dd3 = st.columns(3)
                    dd1.metric("SARIMA RMSE", f'{s.get("rmse", float("nan")):.2f}')
                    dd2.metric("SARIMA MAPE (%)", f'{s.get("mape", float("nan")):.2f}')
                    dd3.metric("SARIMA R²", f'{s.get("r2", float("nan")):.2f}')

                    # En iyi Prophet parametreleri
                    if p.get("best_params"):
                        st.json({"Prophet best_params": p["best_params"]})
                else:
                    st.info("Fonksiyon metrik sözlüğü döndürmedi; konsol çıktısına bakın.")
            except Exception as e:
                st.error(f"Zaman serisi çalışırken hata: {e}")

else:
    st.info("Soldan veri kaynağını seçip **Veriyi yükle ve temizle** düğmesine basın.")
