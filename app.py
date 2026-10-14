import io
import re

import pandas as pd
import streamlit as st


SCALE_PATTERN = re.compile(r"^(WOM|PQ|BT|CS|BI|RI)(?=\d|[_\s-]|$)", re.IGNORECASE)
STATISTIC_COLUMNS = [
    "Ölçek",
    "Madde",
    "Ortalama",
    "Standart sapma",
    "Minimum",
    "Maksimum",
    "Likert (1–5)",
    "Geçersiz yanıt sayısı",
]


def detect_scale(column_name):
    match = SCALE_PATTERN.match(str(column_name).strip())
    return match.group(1).upper() if match else None


def calculate_item_statistics(data):
    statistics = []
    for column in data.columns:
        scale = detect_scale(column)
        if scale is None:
            continue

        values = pd.to_numeric(data[column], errors="coerce")
        nonnumeric_count = int((data[column].notna() & values.isna()).sum())
        responses = values.dropna()
        out_of_range_count = int((~responses.between(1, 5)).sum())
        invalid_count = nonnumeric_count + out_of_range_count
        likert_status = (
            "Yanıt yok"
            if responses.empty and nonnumeric_count == 0
            else "Uygun"
            if invalid_count == 0
            else "Kontrol gerekli"
        )

        statistics.append(
            {
                "Ölçek": scale,
                "Madde": column,
                "Ortalama": responses.mean(),
                "Standart sapma": responses.std(),
                "Minimum": responses.min(),
                "Maksimum": responses.max(),
                "Likert (1–5)": likert_status,
                "Geçersiz yanıt sayısı": invalid_count,
            }
        )

    return pd.DataFrame(statistics, columns=STATISTIC_COLUMNS)


st.set_page_config(
    page_title="PLS-SEM Analytics",
    page_icon="📊",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2.5rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f5f8fc;
        border: 1px solid #e7edf5;
        padding: 1rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PLS-SEM Analytics")
st.caption("Akademik araştırmalar için veri yükleme ve ön inceleme")
st.info(
    "Excel (.xlsx) veya CSV dosyanızı yükleyerek veri setinizi inceleyin. "
    "Bu modül PLS-SEM hesaplaması yapmaz."
)

uploaded_file = st.file_uploader(
    "Veri dosyası seçin",
    type=["xlsx", "csv"],
    help="Excel dosyalarında analiz etmek istediğiniz çalışma sayfasını seçebilirsiniz.",
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".xlsx"):
            workbook = pd.ExcelFile(io.BytesIO(file_bytes))
            sheet_names = workbook.sheet_names
            if not sheet_names:
                st.error("Excel dosyasında çalışma sayfası bulunamadı.")
                st.stop()

            default_sheet = (
                sheet_names.index("Veri_420") if "Veri_420" in sheet_names else 0
            )
            selected_sheet = st.selectbox(
                "Analiz sayfası",
                options=sheet_names,
                index=default_sheet,
            )
            data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected_sheet)
        else:
            data = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    except Exception as error:
        st.error(f"Dosya okunamadı. Lütfen geçerli bir Excel veya CSV dosyası seçin. ({error})")
        st.stop()

    st.subheader("Veri seti özeti")
    participants, variables = data.shape
    missing_values = int(data.isna().sum().sum())
    metric_columns = st.columns(3)
    metric_columns[0].metric("Katılımcı sayısı", f"{participants:,}")
    metric_columns[1].metric("Değişken sayısı", f"{variables:,}")
    metric_columns[2].metric("Eksik veri sayısı", f"{missing_values:,}")

    statistics = calculate_item_statistics(data)
    st.subheader("Madde temel istatistikleri")
    if statistics.empty:
        st.warning(
            "PQ, BT, CS, BI, RI veya WOM ölçeklerine ait madde adı bulunamadı."
        )
    else:
        st.caption(
            "Ölçekler madde adlarının başındaki PQ, BT, CS, BI, RI ve WOM "
            "kodlarına göre tanınır. Likert kontrolü yanıtların 1–5 aralığında "
            "olduğunu denetler."
        )
        st.dataframe(statistics, use_container_width=True, hide_index=True)
        st.download_button(
            "Temel istatistikleri CSV olarak indir",
            data=statistics.to_csv(index=False).encode("utf-8-sig"),
            file_name="pls_sem_temel_istatistikler.csv",
            mime="text/csv",
        )

    with st.expander("Veri önizlemesi"):
        st.dataframe(data.head(20), use_container_width=True)
