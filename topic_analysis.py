import re

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
TARGET_TOPICS = 10

TURKISH_STOPWORDS = """acaba ama aslında az bazı belki biri birkaç birşey biz bu çok çünkü da daha de defa diye eğer en gibi hem hep hepsi her hiç için ile ise kez ki kim mı mu mü mi nasıl ne neden nerede nerde nereye niçin niye o sanki şey siz şu tüm ve veya ya yani ben sen onlar bunlar şunlar bir iki üç ise olan olarak oldu olduğu olur olmuş var yok değil kadar sonra önce ancak fakat lakin ile gibi göre karşı rağmen üzere dolayı bile ise mi bana beni bize bizi sana seni size sizi ona onu onun bunu bunun şunu bunda orada burada şimdi hala hâlâ artık zaten sadece yalnızca bütün tamamen oldukça biraz fazla çokça gayet hiçbir herhangi bence bizim sizin benim senin onların ediyor etti edildi yapıldı yapılıyor yaptı""".split()

CLUSTER_PARAM_SETS = [
    (10, 3),
    (8, 3),
    (7, 2),
    (6, 2),
    (5, 2),
]


def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def clean_comments(series):
    s = series.dropna().astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return s[s != ""]


def compute_embeddings(embedder, docs):
    return np.asarray(
        embedder.encode(docs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    )


def build_model(min_cluster_size, min_samples):
    return BERTopic(
        embedding_model=None,
        umap_model=UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            ngram_range=(1, 2), stop_words=TURKISH_STOPWORDS, lowercase=True
        ),
        ctfidf_model=None,  # BERTopic varsayılanı c-TF-IDF
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )


def _n_topics(model):
    return len([t for t in model.get_topic_info()["Topic"] if t != -1])


def run_analysis(docs, embeddings, progress=lambda msg, frac: None):
    """Doğal konuları bulur; 10'dan fazlaysa reduce_topics ile 10'a indirir."""
    docs = list(docs)
    best, notes = None, []
    for i, (mcs, ms) in enumerate(CLUSTER_PARAM_SETS):
        progress(f"BERTopic çalışıyor (min_cluster_size={mcs}, min_samples={ms})", 0.3 + 0.1 * i)
        model = build_model(mcs, ms)
        topics, _ = model.fit_transform(docs, embeddings)
        n = _n_topics(model)
        notes.append(f"min_cluster_size={mcs}, min_samples={ms}: {n} doğal konu")
        if best is None or n > best[2]:
            best = (model, topics, n, mcs, ms)
        if n >= TARGET_TOPICS:
            break
    model, topics, natural, mcs, ms = best
    reduced = False
    if natural > TARGET_TOPICS:
        progress("Konular 10'a indirgeniyor (reduce_topics)", 0.85)
        for nr in (TARGET_TOPICS, TARGET_TOPICS + 1):
            model.reduce_topics(docs, nr_topics=nr)
            if _n_topics(model) == TARGET_TOPICS:
                break
        topics = list(model.topics_)
        reduced = True
    return {
        "model": model,
        "topics": np.array(topics),
        "natural_topics": natural,
        "final_topics": _n_topics(model),
        "reduced": reduced,
        "params": (mcs, ms),
        "log": notes,
    }


def suggest_name(keywords):
    return " / ".join(k.title() for k in keywords[:3])


def topic_table(model, docs, topics, embeddings=None):
    docs = np.array(docs, dtype=object)
    rows = []
    for t in sorted(set(topics) - {-1}):
        kws = [w for w, _ in model.get_topic(t)][:10]
        idx = np.where(topics == t)[0]
        rep = list(model.get_representative_docs(t) or [])[:5]
        if len(rep) < 5:
            rep += [d for d in docs[idx] if d not in rep][: 5 - len(rep)]
        rows.append(
            {
                "Konu": int(t),
                "Önerilen ad": suggest_name(kws),
                "Anahtar kelimeler": kws,
                "Yorum sayısı": int(len(idx)),
                "Temsilî yorumlar": rep[:5],
            }
        )
    return sorted(rows, key=lambda r: -r["Yorum sayısı"])


def build_excel(docs, topics, table, names):
    import io

    name_map = {r["Konu"]: names[r["Konu"]] for r in table}
    kw_map = {r["Konu"]: ", ".join(r["Anahtar kelimeler"]) for r in table}
    comments = pd.DataFrame(
        {
            "Yorum": list(docs),
            "Konu No": [int(t) for t in topics],
            "Konu Adı": [name_map.get(int(t), "Aykırı (gürültü)") for t in topics],
        }
    )
    summary = pd.DataFrame(
        [
            {
                "Konu No": r["Konu"],
                "Konu Adı": name_map[r["Konu"]],
                "Anahtar Kelimeler": kw_map[r["Konu"]],
                "Yorum Sayısı": r["Yorum sayısı"],
            }
            for r in table
        ]
    )
    outliers = comments[comments["Konu No"] == -1][["Yorum"]]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        comments.to_excel(w, sheet_name="Yorumlar", index=False)
        summary.to_excel(w, sheet_name="Konular", index=False)
        outliers.to_excel(w, sheet_name="Aykırı Yorumlar", index=False)
    return buf.getvalue()
