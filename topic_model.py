"""BERTopic tabanlı konu modelleme işlevleri (Türkçe belediye yorumları)."""
import io
import random

import numpy as np
import pandas as pd

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
TARGET_TOPICS = 10

TURKISH_STOPWORDS = """acaba ama aslında az bazı belki biri birkaç birşey biz bu çok çünkü da daha de defa
diye eğer en gibi hem hep hepsi her hiç için ile ise kez ki kim mı mi mu mü nasıl ne neden nerede nerde
nereye niçin niye o sanki şey siz şu tüm ve veya ya yani ben sen onlar bunlar şunlar bir iki üç ben
bana beni benim sana seni senin ona onu onun bize bizi bizim size sizi sizin onlara onları onların
olarak olan olduğu oldu olur olmuş olsa olmak var yok değil idi imiş ise mi bile kadar sonra önce
ancak fakat lakin ya da ayrıca ise üzere göre rağmen karşı dolayı artık şimdi hala hâlâ zaten sadece
yalnızca bunu bunun buna bunda bundan şunu şuna şundan onda ondan orada burada şurada gerek gerekir
etti ettik etmek edildi edilir yapıldı yapılır yaptı yaptık yapmak bütün tamamen oldukça gayet pek
çoğu bazen sürekli herkes hiçbir kendi kendisi mı mu ben de da ki dır dir dur dür tır tir""".split()

# Denenecek makul kümeleme ayarları (ilki gereksinimdeki varsayılandır).
CLUSTER_SETTINGS = [
    {"min_cluster_size": 10, "min_samples": 3},
    {"min_cluster_size": 8, "min_samples": 2},
    {"min_cluster_size": 6, "min_samples": 2},
    {"min_cluster_size": 5, "min_samples": 1},
]


def read_table(file, sheet=None):
    name = getattr(file, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, sheet_name=sheet or 0)
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    for enc in ("utf-8-sig", "utf-8", "cp1254", "iso-8859-9"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def clean_comments(series):
    """Boş yorumları atar; Türkçe karakterleri korur. (özgün_indeks, metin) döndürür."""
    text = series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    text = text.where(text.notna() & (text != "") & (text.str.lower() != "nan"))
    text = text.dropna()
    return text.astype(str)


def basic_stats(total, comments):
    lengths = comments.str.split().str.len()
    return {
        "Toplam satır": int(total),
        "Boş/geçersiz yorum": int(total - len(comments)),
        "Analiz edilebilir yorum": int(len(comments)),
        "Ortalama kelime sayısı": round(float(lengths.mean()), 1) if len(lengths) else 0,
        "En kısa (kelime)": int(lengths.min()) if len(lengths) else 0,
        "En uzun (kelime)": int(lengths.max()) if len(lengths) else 0,
    }


def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed(embedder, docs):
    return embedder.encode(docs, batch_size=32, show_progress_bar=False)


def _build_model(embedder, min_cluster_size, min_samples, min_df):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=SEED)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            metric="euclidean", cluster_selection_method="eom",
                            prediction_data=True)
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=TURKISH_STOPWORDS,
                                 min_df=min_df, lowercase=True)
    return BERTopic(embedding_model=embedder, umap_model=umap_model,
                    hdbscan_model=hdbscan_model, vectorizer_model=vectorizer,
                    ctfidf_model=ClassTfidfTransformer(), top_n_words=10,
                    calculate_probabilities=False, verbose=False)


def _n_topics(model):
    return len([t for t in model.get_topic_info()["Topic"] if t != -1])


def run_bertopic(docs, embeddings, embedder, progress=lambda f, m: None):
    """Doğal konuları bulur; >10 ise reduce_topics ile 10'a indirir.

    Döndürür: (model, topics, bilgi sözlüğü)
    """
    random.seed(SEED)
    np.random.seed(SEED)
    docs = list(docs)
    best = None
    tried = []
    for i, cfg in enumerate(CLUSTER_SETTINGS):
        progress(0.35 + 0.35 * i / len(CLUSTER_SETTINGS),
                 f"Kümeleme deneniyor (min_cluster_size={cfg['min_cluster_size']}, "
                 f"min_samples={cfg['min_samples']})...")
        min_df = 2 if len(docs) >= 50 else 1
        model = _build_model(embedder, cfg["min_cluster_size"], cfg["min_samples"], min_df)
        topics, _ = model.fit_transform(docs, embeddings)
        n = _n_topics(model)
        tried.append({**cfg, "doğal konu sayısı": n})
        if best is None or n > best[0]:
            best = (n, model, topics, cfg)
        if n >= TARGET_TOPICS:
            break
    natural, model, topics, cfg = best
    info = {"denemeler": tried, "ayar": cfg, "doğal_konu_sayısı": natural, "azaltıldı": False}
    if natural > TARGET_TOPICS:
        progress(0.75, f"{natural} doğal konu 10 konuya indiriliyor (reduce_topics)...")
        model.reduce_topics(docs, nr_topics=TARGET_TOPICS)
        topics = list(model.topics_)
        n = _n_topics(model)
        # BERTopic gürültüyü sayıma katarsa bir ek azaltma gerekebilir.
        if n > TARGET_TOPICS:
            model.reduce_topics(docs, nr_topics=TARGET_TOPICS - 1)
            topics = list(model.topics_)
        info["azaltıldı"] = True
    info["nihai_konu_sayısı"] = _n_topics(model)
    return model, list(topics), info


def suggest_name(keywords, n=3):
    return " / ".join(k.title() for k in keywords[:n]) if keywords else "Adsız konu"


def topic_table(model, topics, embeddings, docs):
    """Konu özetleri ve temsilî yorumlar."""
    topics = np.asarray(topics)
    emb = np.asarray(embeddings)
    rows, reps = [], {}
    for tid in sorted(t for t in set(topics) if t != -1):
        kws = [w for w, _ in model.get_topic(tid)][:10]
        idx = np.where(topics == tid)[0]
        centroid = emb[idx].mean(axis=0)
        sims = emb[idx] @ centroid / (np.linalg.norm(emb[idx], axis=1)
                                      * np.linalg.norm(centroid) + 1e-12)
        top = idx[np.argsort(-sims)[:5]]
        reps[tid] = [docs[i] for i in top]
        rows.append({"Konu No": int(tid), "Yorum sayısı": int(len(idx)),
                     "Anahtar kelimeler": ", ".join(kws), "Önerilen ad": suggest_name(kws)})
    df = pd.DataFrame(rows).sort_values("Yorum sayısı", ascending=False).reset_index(drop=True)
    return df, reps


def to_excel(comments_df, summary_df):
    """comments_df: Yorum, Konu No, Konu adı. summary_df: konu özet tablosu."""
    out = io.BytesIO()
    outliers = comments_df[comments_df["Konu No"] == -1]
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        comments_df.to_excel(w, "Yorum-Konu Atamaları", index=False)
        summary_df.to_excel(w, "Konular", index=False)
        summary_df[["Konu No", "Konu adı", "Yorum sayısı"]].to_excel(
            w, "Konu Başına Yorum Sayısı", index=False)
        outliers.to_excel(w, "Aykırı Yorumlar", index=False)
    return out.getvalue()
