import hashlib
import io

import pandas as pd
import streamlit as st

from pls_sem import (
    DEFAULT_PATHS,
    detect_scale,
    estimate_measurement_model,
    find_out_of_range_items,
    get_measurement_items,
)

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
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 1rem;
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PLS-SEM Analytics")
st.caption("Akademik araştırmalar için veri inceleme ve PLS-SEM ölçüm modeli analizi")
st.info(
    "Excel (.xlsx) veya CSV dosyanızı yükleyerek veri setinizi inceleyin ve "
    "yansıtıcı Mode A ölçüm modeli analizi yapın."
)

uploaded_file = st.file_uploader(
    "Veri dosyası seçin",
    type=["xlsx", "csv"],
    help="Excel dosyalarında analiz etmek istediğiniz çalışma sayfasını seçebilirsiniz.",
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        selected_sheet = None
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
        out_of_range_items = find_out_of_range_items(data)
        if out_of_range_items:
            st.warning(
                "1–5 aralığı dışında yanıt içeren maddeler: "
                + ", ".join(map(str, out_of_range_items))
                + ". Analizden önce bu yanıtları kontrol edin."
            )
        st.download_button(
            "Temel istatistikleri CSV olarak indir",
            data=statistics.to_csv(index=False).encode("utf-8-sig"),
            file_name="pls_sem_temel_istatistikler.csv",
            mime="text/csv",
        )

    scale_items = get_measurement_items(data)
    st.subheader("PLS-SEM ölçüm modeli")
    if not scale_items:
        st.warning("Analiz için PQ, BT, CS, BI, RI veya WOM maddesi bulunamadı.")
    else:
        scales = list(scale_items)
        st.caption(
            "Yalnızca tanınan ölçek maddeleri analize alınır; katılımcı kimliği ve "
            "diğer sütunlar analiz dışında kalır. Eksik yanıt içeren satırlar "
            "hesaplama sırasında çıkarılır."
        )
        available_paths = [
            (source, target)
            for source in scales
            for target in scales
            if source != target
        ]
        default_paths = [
            path for path in DEFAULT_PATHS if path[0] in scales and path[1] in scales
        ]
        path_labels = {
            f"{source} → {target}": (source, target)
            for source, target in available_paths
        }
        selected_labels = st.multiselect(
            "Gizil değişkenler arasındaki yönlü ilişkiler",
            options=list(path_labels),
            default=[
                label for label, path in path_labels.items() if path in default_paths
            ],
            help=(
                "Varsayılan modelde PQ ve BT, CS ve BI'yi; CS ve BI ise RI ve "
                "WOM'u yordar."
            ),
        )
        selected_paths = [path_labels[label] for label in selected_labels]
        analysis_signature = (
            hashlib.sha256(file_bytes).hexdigest(),
            selected_sheet,
            tuple(selected_paths),
        )
        if st.button("Ölçüm modelini hesapla", type="primary"):
            try:
                results = estimate_measurement_model(data, selected_paths)
                st.session_state["measurement_model_result"] = (
                    analysis_signature,
                    results,
                    None,
                )
            except ValueError as error:
                st.session_state["measurement_model_result"] = (
                    analysis_signature,
                    None,
                    str(error),
                )
            except Exception as error:
                st.session_state["measurement_model_result"] = (
                    analysis_signature,
                    None,
                    f"Analiz sırasında beklenmeyen bir hesaplama hatası oluştu: {error}",
                )
        stored_result = st.session_state.get("measurement_model_result")
        if stored_result and stored_result[0] == analysis_signature:
            results, error_message = stored_result[1:]
            if error_message:
                st.error(error_message)
            elif results is not None:
                st.success(
                    f"Analiz tamamlandı: {results.observations:,} katılımcı kullanıldı; "
                    f"{results.dropped_rows:,} eksik yanıtlı satır çıkarıldı. "
                    f"PLS algoritması yakınsadı: {'Evet' if results.converged else 'Hayır'}."
                )
                st.caption(
                    "Yüklerin yönü, her yapıda yüklerin toplamı pozitif olacak "
                    "şekilde belirlenmiştir; yol katsayıları aynı yönlendirmeye "
                    "göre dönüştürülür. HTMT tablosu klasik HTMT yöntemini kullanır."
                )
                result_tables = {
                    "Dış yükler": results.loadings,
                    "Güvenirlik": results.reliability,
                    "HTMT": results.validity,
                    "Yapısal yollar": results.paths,
                }
                tabs = st.tabs(list(result_tables))
                for tab, (title, result_table) in zip(tabs, result_tables.items()):
                    with tab:
                        st.dataframe(
                            result_table, use_container_width=True, hide_index=True
                        )
                        st.download_button(
                            f"{title} sonuçlarını CSV olarak indir",
                            data=result_table.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"pls_sem_{title.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"download_{title}",
                        )
                excel_file = io.BytesIO()
                with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                    for title, result_table in result_tables.items():
                        result_table.to_excel(
                            writer, sheet_name=title[:31], index=False
                        )
                st.download_button(
                    "Tüm PLS-SEM sonuçlarını Excel olarak indir",
                    data=excel_file.getvalue(),
                    file_name="pls_sem_sonuclari.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with st.expander("Veri önizlemesi"):
        st.dataframe(data.head(20), use_container_width=True)
