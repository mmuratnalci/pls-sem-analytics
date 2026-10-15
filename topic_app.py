import traceback

import pandas as pd
import plotly.express as px
import streamlit as st

import topic_model as tm

st.set_page_config(page_title="Belediye Hizmet Binası – Yorum Analizi", layout="wide")
st.title("Belediye Hizmet Binası – Yapay Zekâ Destekli Yorum Analizi")
st.caption("BERTopic ile Türkçe yorumlardan konu keşfi. Konular anahtar kelime sayımıyla "
           "değil, gömme + UMAP + HDBSCAN + c-TF-IDF ile bulunur.")


@st.cache_resource(show_spinner="Gömme modeli yükleniyor...")
def get_embedder():
    return tm.load_embedder()


@st.cache_data(show_spinner=False)
def cached_embeddings(docs):
    return tm.embed(get_embedder(), list(docs))


st.header("1. Veri girişi")
upload = st.file_uploader("Excel (.xlsx) veya CSV dosyası", type=["xlsx", "xls", "csv"])
if upload is None:
    st.info("Başlamak için yorumları içeren bir dosya yükleyin.")
    st.stop()

try:
    sheet = None
    if upload.name.lower().endswith((".xlsx", ".xls")):
        sheets = pd.ExcelFile(upload).sheet_names
        sheet = st.selectbox("Excel sayfası", sheets)
        upload.seek(0)
    df = tm.read_table(upload, sheet)
except Exception as exc:
    st.error(f"Dosya okunamadı: {exc}")
    st.stop()

if df.empty:
    st.error("Dosyada veri bulunamadı.")
    st.stop()

text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
options = text_cols or list(df.columns)
col = st.selectbox("Yorum sütunu", options)
comments = tm.clean_comments(df[col])
stats = tm.basic_stats(len(df), comments)
cols = st.columns(len(stats))
for c, (k, v) in zip(cols, stats.items()):
    c.metric(k, v)
if len(comments) < 30:
    st.error("Anlamlı bir konu analizi için en az 30 dolu yorum gerekir.")
    st.stop()

st.header("2. Analiz")
if st.button("BERTopic Analizini Başlat", type="primary"):
    bar = st.progress(0.0, text="Başlıyor...")

    def progress(f, m):
        bar.progress(min(f, 1.0), text=m)

    try:
        docs = comments.tolist()
        progress(0.05, "Gömme modeli hazırlanıyor...")
        embedder = get_embedder()
        progress(0.15, "Yorumlar vektöre dönüştürülüyor (CPU, birkaç dakika sürebilir)...")
        emb = cached_embeddings(tuple(docs))
        model, topics, info = tm.run_bertopic(docs, emb, embedder, progress)
        progress(0.9, "Sonuçlar hazırlanıyor...")
        summary, reps = tm.topic_table(model, topics, emb, docs)
        st.session_state["res"] = dict(model=model, topics=topics, info=info, summary=summary,
                                       reps=reps, docs=docs, total=len(df))
        progress(1.0, "Analiz tamamlandı.")
    except Exception as exc:
        bar.empty()
        st.error(f"Analiz sırasında hata oluştu: {exc}")
        with st.expander("Teknik ayrıntı"):
            st.code(traceback.format_exc())

res = st.session_state.get("res")
if not res:
    st.stop()

model, topics, info, summary = res["model"], res["topics"], res["info"], res["summary"]
n_out = sum(1 for t in topics if t == -1)
n_topics = len(summary)

st.header("3. Sonuçlar")
m = st.columns(4)
m[0].metric("Toplam yorum", res["total"])
m[1].metric("Analiz edilen yorum", len(topics))
m[2].metric("Keşfedilen konu", n_topics)
m[3].metric("Aykırı yorum", n_out)

st.caption(f"Kullanılan ayar: min_cluster_size={info['ayar']['min_cluster_size']}, "
           f"min_samples={info['ayar']['min_samples']}. Doğal konu sayısı: "
           f"{info['doğal_konu_sayısı']}.")
if info["azaltıldı"]:
    st.info(f"{info['doğal_konu_sayısı']} doğal konu, BERTopic reduce_topics ile 10 konuya indirildi.")
if n_topics < tm.TARGET_TOPICS:
    st.warning(f"Veriden yalnızca {n_topics} anlamlı konu çıkarılabildi (10'dan az). "
               "Farklı kümeleme ayarları denendi; konu uydurulmadı veya yapay olarak bölünmedi.")
    st.dataframe(pd.DataFrame(info["denemeler"]), hide_index=True)

st.subheader("Konu adları")
st.caption("Adlar anahtar kelimelerden önerilmiştir; düzenleyebilirsiniz.")
edited = st.data_editor(
    summary[["Konu No", "Önerilen ad", "Yorum sayısı", "Anahtar kelimeler"]],
    disabled=["Konu No", "Yorum sayısı", "Anahtar kelimeler"], hide_index=True,
    use_container_width=True, key="names")
names = dict(zip(edited["Konu No"], edited["Önerilen ad"].fillna("").astype(str)))
summary = summary.assign(**{"Konu adı": summary["Konu No"].map(names)})
label = {t: f"{names[t]} (#{t})" for t in names}

chart = summary.assign(Etiket=summary["Konu No"].map(label)).sort_values("Yorum sayısı")
fig = px.bar(chart, x="Yorum sayısı", y="Etiket", orientation="h", title="Konulara göre yorum sayısı")
fig.update_layout(yaxis_title="")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Temsilî yorumlar")
for _, r in summary.iterrows():
    with st.expander(f"{r['Konu adı']} – {r['Yorum sayısı']} yorum"):
        st.write("**Anahtar kelimeler:** " + r["Anahtar kelimeler"])
        for d in res["reps"][r["Konu No"]]:
            st.markdown(f"- {d}")

st.subheader("BERTopic görselleştirmeleri")
for title, fn in [("Konu benzerliği (konular arası mesafe)", lambda: model.visualize_topics()),
                  ("Konu anahtar kelimeleri", lambda: model.visualize_barchart(
                      top_n_topics=max(n_topics, 1), n_words=10)),
                  ("Konu benzerlik ısı haritası", lambda: model.visualize_heatmap())]:
    try:
        st.markdown(f"**{title}**")
        st.plotly_chart(fn(), use_container_width=True)
    except Exception as exc:
        st.warning(f"'{title}' oluşturulamadı: {exc}")

st.header("4. Excel çıktısı")
out = pd.DataFrame({"Yorum": res["docs"], "Konu No": topics})
out["Konu adı"] = out["Konu No"].map(names).fillna("Aykırı (konu yok)")
xlsx = tm.to_excel(out, summary[["Konu No", "Konu adı", "Anahtar kelimeler", "Yorum sayısı"]])
st.download_button("Excel dosyasını indir", xlsx, "bertopic_sonuclari.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
