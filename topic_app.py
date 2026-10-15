import io

import pandas as pd
import plotly.express as px
import streamlit as st

import topic_analysis as ta

st.set_page_config(page_title="Belediye Hizmet Binası – Yorum Analizi", layout="wide")


@st.cache_resource(show_spinner=False)
def get_embedder():
    return ta.load_embedder()


@st.cache_data(show_spinner=False)
def get_embeddings(docs):
    return ta.compute_embeddings(get_embedder(), list(docs))


st.title("Belediye Hizmet Binası – Yapay Zekâ Destekli Yorum Analizi")
st.caption("BERTopic (c-TF-IDF) ile konu modelleme. Tüm sonuçlar modelden üretilir; tekrarlanabilirlik için sabit tohum (42) kullanılır.")

file = st.file_uploader("Excel (.xlsx) veya CSV dosyası yükleyin", type=["xlsx", "csv"])
raw = None
if file is not None:
    try:
        if file.name.lower().endswith(".csv"):
            raw = pd.read_csv(file)
        else:
            sheets = pd.ExcelFile(file).sheet_names
            sheet = st.selectbox("Excel sayfası", sheets)
            raw = pd.read_excel(file, sheet_name=sheet)
    except Exception as e:
        st.error(f"Dosya okunamadı: {e}")

if raw is not None and len(raw.columns):
    col = st.selectbox("Yorum sütunu", list(raw.columns))
    clean = ta.clean_comments(raw[col])
    lengths = clean.str.len()
    c = st.columns(4)
    c[0].metric("Toplam satır", len(raw))
    c[1].metric("Geçerli yorum", len(clean))
    c[2].metric("Boş/atılan", len(raw) - len(clean))
    c[3].metric("Ort. karakter", int(lengths.mean()) if len(clean) else 0)

    if st.button("BERTopic Analizini Başlat", type="primary"):
        if len(clean) < 30:
            st.error("Analiz için en az 30 geçerli yorum gerekir.")
        else:
            try:
                bar = st.progress(0.0)
                status = st.empty()

                def progress(msg, frac):
                    status.info(msg)
                    bar.progress(min(frac, 1.0))

                progress("Gömme modeli yükleniyor ve yorumlar vektörleştiriliyor", 0.1)
                docs = tuple(clean.tolist())
                emb = get_embeddings(docs)
                res = ta.run_analysis(docs, emb, progress)
                bar.progress(1.0)
                status.success("Analiz tamamlandı.")
                st.session_state["res"] = (res, docs)
            except Exception as e:
                st.error(f"Analiz sırasında hata oluştu: {e}")

if "res" in st.session_state:
    res, docs = st.session_state["res"]
    model, topics = res["model"], res["topics"]
    table = ta.topic_table(model, docs, topics)
    n_out = int((topics == -1).sum())

    c = st.columns(4)
    c[0].metric("Toplam yorum", len(docs))
    c[1].metric("Analiz edilen yorum", len(docs))
    c[2].metric("Keşfedilen konu", res["final_topics"])
    c[3].metric("Aykırı yorum", n_out)

    st.caption("Denenen kümeleme ayarları: " + " | ".join(res["log"]))
    if res["natural_topics"] < ta.TARGET_TOPICS:
        st.warning(
            f"Veriden yalnızca {res['natural_topics']} doğal konu çıkarılabildi (10'dan az). "
            "Farklı kümeleme parametreleri denendi; konu uydurulmadı ve yapay bölme yapılmadı."
        )
    elif res["reduced"]:
        st.info(f"{res['natural_topics']} doğal konu reduce_topics ile 10'a indirildi.")

    st.subheader("Konu adları (düzenlenebilir)")
    names = {}
    for r in table:
        names[r["Konu"]] = st.text_input(
            f"Konu {r['Konu']}", r["Önerilen ad"], key=f"name_{r['Konu']}"
        )

    df = pd.DataFrame(
        {
            "Konu": [names[r["Konu"]] for r in table],
            "Yorum sayısı": [r["Yorum sayısı"] for r in table],
        }
    )
    fig = px.bar(df.sort_values("Yorum sayısı"), x="Yorum sayısı", y="Konu", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    for r in table:
        with st.expander(f"{names[r['Konu']]} ({r['Yorum sayısı']} yorum)"):
            st.write("**Anahtar kelimeler:** " + ", ".join(r["Anahtar kelimeler"]))
            for d in r["Temsilî yorumlar"]:
                st.markdown(f"- {d}")

    st.subheader("BERTopic görselleştirmeleri")
    try:
        st.plotly_chart(model.visualize_topics(), use_container_width=True)
    except Exception as e:
        st.info(f"Konu benzerliği haritası oluşturulamadı: {e}")
    try:
        st.plotly_chart(model.visualize_barchart(top_n_topics=10, n_words=10), use_container_width=True)
    except Exception as e:
        st.info(f"Anahtar kelime grafiği oluşturulamadı: {e}")

    st.download_button(
        "Excel çıktısını indir",
        ta.build_excel(docs, topics, table, names),
        file_name="bertopic_sonuclar.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
