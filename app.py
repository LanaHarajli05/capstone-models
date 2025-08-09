import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, classification_report, confusion_matrix
)

# -------------------------
# Page & cache
# -------------------------
st.set_page_config(page_title="🎓 Hub Analytics — EDA & Predictions", layout="wide")
st.title("🎓 Hub Analytics — EDA & Predictions")
st.caption("Explore your data, and predict **Final Status** and **Major Grouping** for future/unknown students.")

@st.cache_data(show_spinner=False)
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.replace("\xa0"," ").strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def _pick_col(df: pd.DataFrame, candidates):
    m = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in m: return m[key]
    return ""

def _build_preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, num, cat

def _impute_simple(X: pd.DataFrame, num_cols, cat_cols):
    X = X.copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown")
    return X

def _plot_corr(df_num: pd.DataFrame, title="Correlation Matrix"):
    if df_num.shape[1] == 0:
        st.info("No numeric columns to correlate.")
        return
    corr = df_num.corr(numeric_only=True)
    fig = plt.figure()
    plt.imshow(corr, interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(title); plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def _plot_histograms(df_num: pd.DataFrame, bins=20):
    for col in df_num.columns:
        fig = plt.figure()
        plt.hist(df_num[col].dropna(), bins=bins)
        plt.title(f"Histogram — {col}")
        plt.xlabel(col); plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)

def _plot_bars(df: pd.DataFrame, cat_cols, top_n=20):
    for c in cat_cols:
        vc = df[c].fillna("Unknown").value_counts().head(top_n)
        fig = plt.figure()
        plt.bar(vc.index.astype(str), vc.values)
        plt.title(f"Bar Chart — {c}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)

def _plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def _topk_from_proba(proba, classes, k=3):
    labels, probs = [], []
    for row in proba:
        idx = np.argsort(row)[::-1][:k]
        labels.append(", ".join(str(classes[i]) for i in idx))
        probs.append(", ".join(f"{row[i]:.2f}" for i in idx))
    return labels, probs

def _show_metrics(title, y_true, y_pred):
    st.subheader(title)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision (macro)", f"{p:.3f}")
    c3.metric("Recall (macro)", f"{r:.3f}")
    c4.metric("F1 (macro)", f"{f1:.3f}")
    st.text("Classification Report")
    st.code(classification_report(y_true, y_pred, zero_division=0), language="text")

def _train_model(df, target, id_col, model_key, test_size, seed, feature_whitelist=None):
    # select features (avoid leakage: no target, no id)
    all_feats = [c for c in df.columns if c != target]
    if id_col and id_col in all_feats:
        all_feats.remove(id_col)
    feats = [c for c in all_feats if (feature_whitelist is None or c in feature_whitelist)]

    y = df[target].astype(str)
    X = df[feats].copy()

    num0 = X.select_dtypes(include=["number"]).columns.tolist()
    cat0 = [c for c in X.columns if c not in num0]
    X = _impute_simple(X, num0, cat0)

    pre, _, _ = _build_preprocessor(X)

    if model_key == "LR":
        clf = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=3000)
    elif model_key == "RF":
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    elif model_key == "XGB":
        clf = XGBClassifier(
            n_estimators=400, learning_rate=0.08, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError("Unknown model key")

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    strat = y if y.nunique() > 1 else None
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=test_size, random_state=seed, stratify=strat
    )

    pipe.fit(Xtr, ytr)
    yte_pred = pipe.predict(Xte)

    has_proba = hasattr(pipe, "predict_proba")
    proba_full = pipe.predict_proba(X) if has_proba else None
    classes = pipe.classes_ if has_proba else None

    return {
        "pipe": pipe,
        "features": feats,
        "X": X,
        "y": y,
        "idx_te": idx_te,
        "yte": yte,
        "yte_pred": yte_pred,
        "has_proba": has_proba,
        "proba_full": proba_full,
        "classes": classes,
    }

# -------------------------
# Sidebar: upload + filters + settings
# -------------------------
with st.sidebar:
    st.header("1) Upload Data")
    file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

    st.header("2) Modeling Settings")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42)
    top_k = st.slider("Top-K (for Major)", 1, 5, 3)

if not file:
    st.info("⬅️ Upload a dataset to begin.")
    st.stop()

df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
df = _normalize_cols(df)

# Column names (robust)
col_final = _pick_col(df, ["Final Status", "Final Status "])
col_major = _pick_col(df, ["Major Grouping", "Major grouping", "major grouping"])
col_id = _pick_col(df, ["NAME", "Name", "ID", "Id"])

# Standardize major labels
if col_major:
    canon = {
        "engineering": "Engineering","business":"Business","cs/it/mis":"CS/IT/MIS",
        "sciences":"Sciences","health":"Health","arts & others":"Arts & Others",
        "arts and others":"Arts & Others"
    }
    df[col_major] = df[col_major].astype(str).str.strip().apply(lambda v: canon.get(v.lower(), v))

# Filters (dynamic)
candidate_filters = [
    ["Cohort", "Admit Semester", "Admit semester"],
    ["Gender"],
    ["University"],
    ["COR", "Country of Residence"],
    ["Nationality"],
    ["Employment"],
    ["Final Status", "Final Status "],
]
filter_cols = [_pick_col(df, cands) for cands in candidate_filters]
filter_cols = [c for c in filter_cols if c]

with st.sidebar:
    st.header("3) Filters (apply to EDA & metrics/tables)")
    chosen = {}
    for c in filter_cols:
        vals = sorted(df[c].dropna().astype(str).unique().tolist())
        chosen[c] = set(st.multiselect(c, vals, default=[]))

mask = pd.Series(True, index=df.index)
for c, sel in chosen.items():
    if sel:
        mask &= df[c].astype(str).isin(sel)
df_view = df.loc[mask].copy()

# -------------------------
# Tabs
# -------------------------
tab_eda, tab_final, tab_major = st.tabs(["EDA (Interactive)", "Final Status Prediction", "Major Group Prediction"])

# ===== TAB 1: EDA =====
with tab_eda:
    st.markdown("### Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (filtered)", f"{len(df_view):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing values (%)", f"{100*df_view.isna().mean().mean():.1f}")

    st.markdown("### Target Distributions")
    cols_t = st.columns(2)
    if col_final:
        with cols_t[0]:
            st.write("Final Status")
            st.bar_chart(df_view[col_final].value_counts())
    if col_major:
        with cols_t[1]:
            st.write("Major Grouping")
            st.bar_chart(df_view[col_major].value_counts())

    st.divider()
    st.markdown("### Correlation (numeric)")
    _plot_corr(df_view.select_dtypes(include=["number"]))

    st.divider()
    st.markdown("### Histograms (numeric)")
    bins = st.slider("Bins", 5, 60, 20)
    _plot_histograms(df_view.select_dtypes(include=["number"]), bins=bins)

    st.divider()
    st.markdown("### Bar Charts (categorical)")
    cat_cols = [c for c in df_view.columns if df_view[c].dtype == "object" and c != col_id]
    topn = st.slider("Top categories to show", 5, 30, 20)
    _plot_bars(df_view, cat_cols[:8], top_n=topn)  # first 8 to avoid spam

# ===== TAB 2: Final Status Prediction (future students) =====
with tab_final:
    if not col_final:
        st.warning("Final Status column not found.")
    else:
        st.markdown("### Train & Evaluate (historical data)")
        # features available *before* final status exists
        pre_enrollment_feats = [c for c in df.columns if c not in [col_final, col_id, col_major]]
        feats_sel = st.multiselect("Features to use (avoid leakage)", sorted(pre_enrollment_feats), default=sorted(pre_enrollment_feats))

        models = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB"}
        res = {}
        for name, key in models.items():
            r = _train_model(df.dropna(subset=[col_final]), col_final, col_id, key, test_size, seed, feature_whitelist=feats_sel)
            res[name] = r
            st.subheader(f"{name} — Metrics")
            _show_metrics("Test performance", r["yte"], r["yte_pred"])
            st.markdown("**Confusion Matrix**")
            lbls = sorted(df[col_final].dropna().astype(str).unique().tolist())
            _plot_confusion(r["yte"], r["yte_pred"], lbls)

        st.divider()
        st.markdown("### Predict Final Status for a New / Unknown Student")
        m_choice = st.selectbox("Choose trained model", list(models.keys()))
        chosen_model = res[m_choice]

        # compact form
        st.write("Enter fields (unknowns are ok):")
        inputs = {}
        cols = st.columns(2)
        half = (len(chosen_model["features"]) + 1) // 2
        left, right = chosen_model["features"][:half], chosen_model["features"][half:]

        for f in left:
            if df[f].dtype == "object":
                values = ["Unknown"] + sorted(df[f].dropna().astype(str).unique().tolist())
                inputs[f] = cols[0].selectbox(f, values, index=0, key=f"fsL-{f}")
            else:
                med = float(pd.to_numeric(df[f], errors="coerce").median(skipna=True)) if df[f].notna().any() else 0.0
                inputs[f] = cols[0].number_input(f, value=med, key=f"fsLn-{f}")

        for f in right:
            if df[f].dtype == "object":
                values = ["Unknown"] + sorted(df[f].dropna().astype(str).unique().tolist())
                inputs[f] = cols[1].selectbox(f, values, index=0, key=f"fsR-{f}")
            else:
                med = float(pd.to_numeric(df[f], errors="coerce").median(skipna=True)) if df[f].notna().any() else 0.0
                inputs[f] = cols[1].number_input(f, value=med, key=f"fsRn-{f}")

        if st.button("Predict Final Status"):
            X_new = pd.DataFrame({k: [v] for k, v in inputs.items()}, columns=chosen_model["features"])
            if chosen_model["has_proba"]:
                probs = chosen_model["pipe"].predict_proba(X_new)[0]
                classes = chosen_model["classes"]
                idx = np.argsort(probs)[::-1]
                st.success(f"Predicted Final Status: **{classes[idx[0]]}**")
                st.write("Probabilities:")
                for i in idx:
                    st.write(f"- {classes[i]}: {probs[i]:.2f}")
            else:
                pred = chosen_model["pipe"].predict(X_new)[0]
                st.success(f"Predicted Final Status: **{pred}**")

# ===== TAB 3: Major Group Prediction (future students) =====
with tab_major:
    if not col_major:
        st.warning("Major Grouping column not found.")
    else:
        st.markdown("### Train & Evaluate (historical data)")
        pre_enrollment_feats = [c for c in df.columns if c not in [col_major, col_id, col_final]]
        feats_sel_m = st.multiselect("Features to use (avoid leakage)", sorted(pre_enrollment_feats), default=sorted(pre_enrollment_feats))

        models = {"Logistic Regression": "LR", "Random Forest": "RF", "XGBoost": "XGB"}
        resM = {}
        for name, key in models.items():
            r = _train_model(df.dropna(subset=[col_major]), col_major, col_id, key, test_size, seed, feature_whitelist=feats_sel_m)
            resM[name] = r
            st.subheader(f"{name} — Metrics")
            _show_metrics("Test performance", r["yte"], r["yte_pred"])
            st.markdown("**Confusion Matrix**")
            lbls = sorted(df[col_major].dropna().astype(str).unique().tolist())
            _plot_confusion(r["yte"], r["yte_pred"], lbls)

            # Top-K accuracy (business metric)
            if r["has_proba"]:
                proba_te = r["pipe"].predict_proba(r["X"].iloc[r["idx_te"]])
                classes = r["classes"]
                y_true = r["yte"].values
                correct = 0
                for i, row in enumerate(proba_te):
                    idx = np.argsort(row)[::-1][:top_k]
                    if y_true[i] in classes[idx]:
                        correct += 1
                topk_acc = correct / len(y_true) if len(y_true) else 0.0
                st.metric(f"Top-{top_k} Accuracy", f"{topk_acc:.3f}")

        st.divider()
        st.markdown("### Predict Major Group for a New / Unknown Student")
        m_choice = st.selectbox("Choose trained model ", list(models.keys()), key="maj_model_choice")
        chosen = resM[m_choice]

        st.write("Enter fields (unknowns are ok):")
        inputs = {}
        cols = st.columns(2)
        half = (len(chosen["features"]) + 1) // 2
        left, right = chosen["features"][:half], chosen["features"][half:]
        for f in left:
            if df[f].dtype == "object":
                values = ["Unknown"] + sorted(df[f].dropna().astype(str).unique().tolist())
                inputs[f] = cols[0].selectbox(f, values, index=0, key=f"majL-{f}")
            else:
                med = float(pd.to_numeric(df[f], errors="coerce").median(skipna=True)) if df[f].notna().any() else 0.0
                inputs[f] = cols[0].number_input(f, value=med, key=f"majLn-{f}")
        for f in right:
            if df[f].dtype == "object":
                values = ["Unknown"] + sorted(df[f].dropna().astype(str).unique().tolist())
                inputs[f] = cols[1].selectbox(f, values, index=0, key=f"majR-{f}")
            else:
                med = float(pd.to_numeric(df[f], errors="coerce").median(skipna=True)) if df[f].notna().any() else 0.0
                inputs[f] = cols[1].number_input(f, value=med, key=f"majRn-{f}")

        if st.button("Predict Major Group"):
            X_new = pd.DataFrame({k: [v] for k, v in inputs.items()}, columns=chosen["features"])
            if chosen["has_proba"]:
                probs = chosen["pipe"].predict_proba(X_new)[0]
                classes = chosen["classes"]
                idx = np.argsort(probs)[::-1]
                st.success(f"Predicted Major Group: **{classes[idx[0]]}**")
                st.write("Top probabilities:")
                for i in idx[:min(5, len(classes))]:
                    st.write(f"- {classes[i]}: {probs[i]:.2f}")
            else:
                pred = chosen["pipe"].predict(X_new)[0]
                st.success(f"Predicted Major Group: **{pred}**")
