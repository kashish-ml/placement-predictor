import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

data = joblib.load("placement_model.joblib")
model = data["model"]
columns = data["columns"]
info = data["info"]

st.title("🎓 Student Placement Predictor")
st.caption("Data source: " + data["source_name"] + " | " + data["source_url"])

st.sidebar.header("Student details")
row = {}
for col in columns:
    item = info[col]
    if item["type"] == "number":
        row[col] = st.sidebar.slider(col, item["min"], item["max"], item["default"], step=item["step"])
    else:
        row[col] = st.sidebar.selectbox(col, item["options"])


def get_prob(r):
    return model.predict_proba(pd.DataFrame([r], columns=columns))[0][1]


prob = get_prob(row)
st.metric("Placement probability", str(round(prob * 100, 1)) + " %")
st.progress(int(prob * 100))
if prob >= 0.75:
    st.success("High chance of placement")
elif prob >= 0.5:
    st.warning("Moderate chance")
else:
    st.error("Low chance")

number_cols = [c for c in columns if info[c]["type"] == "number"]

st.header("What-if analysis")
pick = st.selectbox("Which feature do you want to change?", number_cols)
values = np.linspace(info[pick]["min"], info[pick]["max"], 20)
probs = []
for v in values:
    r = dict(row)
    r[pick] = float(v)
    probs.append(get_prob(r) * 100)
fig, ax = plt.subplots()
ax.plot(values, probs, marker="o")
ax.axvline(row[pick], color="red", linestyle="--")
ax.set_xlabel(pick)
ax.set_ylabel("Placement probability (%)")
st.pyplot(fig)

st.header("Improvement tips")
CANNOT_CHANGE = ["Age", "Gender", "Hostel"]
tips = []
for col in number_cols:
    if col in CANNOT_CHANGE:
        continue
    step = (info[col]["max"] - info[col]["min"]) * 0.1
    best_gain, best_value = 0, None
    for change in (step, -step):
        new_value = min(max(row[col] + change, info[col]["min"]), info[col]["max"])
        r = dict(row)
        r[col] = new_value
        gain = get_prob(r) - prob
        if gain > best_gain:
            best_gain, best_value = gain, new_value
    if best_value is not None and best_gain > 0.005:
        tips.append((col, row[col], best_value, best_gain))
tips.sort(key=lambda t: t[3], reverse=True)
if tips:
    for col, old, new, gain in tips[:3]:
        st.write("• " + col + ": " + str(round(old, 2)) + " → " + str(round(new, 2))
                 + "  (probability +" + str(round(gain * 100, 1)) + " points)")
else:
    st.write("No clear improvement found right now.")

st.header("Model performance")
for k, v in data["metrics"].items():
    st.write(k + ": " + str(round(v, 3)))
st.write("Best model: " + data["best_name"])
st.dataframe(data["results"])
st.subheader("Feature importance")
st.bar_chart(data["importance"])
st.subheader("Feature importance (detailed chart)")
try:
    st.image("feature_importance.png", caption="Permutation importance: how much the score drops when a feature is shuffled")
except Exception:
    st.info("Run train.py again to create the feature importance image.")

st.header("Why this prediction? (Explainability)")
lr = model.named_steps["model"]
prep_step = model.named_steps["prep"]
if hasattr(lr, "coef_"):
    row_df = pd.DataFrame([row], columns=columns)
    transformed = prep_step.transform(row_df)[0]
    names = list(prep_step.get_feature_names_out())
    coefs = lr.coef_[0]
    contrib = {}
    for name, coef, val in zip(names, coefs, transformed):
        if name.startswith("num__"):
            contrib[name[5:]] = coef * val
    for col in columns:
        if info[col]["type"] == "text":
            prefix = "cat__" + col + "_"
            col_coefs = {n: c for n, c in zip(names, coefs) if n.startswith(prefix)}
            chosen = prefix + str(row[col])
            contrib[col] = col_coefs[chosen] - sum(col_coefs.values()) / len(col_coefs)
    contrib = pd.Series(contrib).sort_values()
    fig2, ax2 = plt.subplots()
    colors = ["green" if v > 0 else "red" for v in contrib.values]
    ax2.barh(contrib.index, contrib.values, color=colors)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("Impact (green = increases probability, red = decreases it)")
    plt.tight_layout()
    st.pyplot(fig2)
    best_plus = contrib.sort_values(ascending=False).head(2)
    best_minus = contrib.sort_values().head(2)
    st.success("Factors improving your placement chances: " + ", ".join(best_plus.index))
    st.warning("Factors reducing your placement chances: " + ", ".join(best_minus.index))
else:
    st.info("This section works only for the Logistic Regression model.")

st.header("Threshold tuning (Precision vs Recall)")
if "test_proba" in data:
    y_true = np.array(data["y_test"])
    p_test = np.array(data["test_proba"])
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, step=0.05)

    def scores_at(t):
        pred_t = (p_test >= t).astype(int)
        tp = int(((pred_t == 1) & (y_true == 1)).sum())
        fp = int(((pred_t == 1) & (y_true == 0)).sum())
        fn = int(((pred_t == 0) & (y_true == 1)).sum())
        tn = int(((pred_t == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1, tp, fp, fn, tn

    prec, rec, f1, tp, fp, fn, tn = scores_at(threshold)
    st.write("Precision: " + str(round(prec, 3)) + " | Recall: " + str(round(rec, 3))
             + " | F1: " + str(round(f1, 3)))
    st.write("Placed students correctly found: " + str(tp) + " | missed: " + str(fn))
    st.write("Not-placed students wrongly marked as placed: " + str(fp))
    if prob >= threshold:
        st.success("At this threshold, the current student is predicted: PLACED")
    else:
        st.error("At this threshold, the current student is predicted: NOT PLACED")

    ts = np.arange(0.05, 0.96, 0.05)
    p_list, r_list, f_list = [], [], []
    for t in ts:
        a, b, c = scores_at(t)[:3]
        p_list.append(a)
        r_list.append(b)
        f_list.append(c)
    fig3, ax3 = plt.subplots()
    ax3.plot(ts, p_list, label="Precision")
    ax3.plot(ts, r_list, label="Recall")
    ax3.plot(ts, f_list, label="F1")
    ax3.axvline(threshold, color="red", linestyle="--")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Score")
    ax3.legend()
    st.pyplot(fig3)
    st.caption("Low threshold = catches more placed students (high recall) but makes more "
               "false alarms. High threshold = fewer false alarms (high precision) but misses more placed students.")
else:
    st.info("Run train.py again to enable threshold tuning.")

st.header("Overfitting check (Learning curve)")
try:
    lc = joblib.load("learning_curve.joblib")
    fig4, ax4 = plt.subplots()
    ax4.plot(lc["sizes"], lc["train"], marker="o", label="Training score")
    ax4.plot(lc["sizes"], lc["val"], marker="o", label="Validation score")
    ax4.set_xlabel("Number of training students")
    ax4.set_ylabel("ROC-AUC")
    ax4.legend()
    st.pyplot(fig4)
    gap = lc["train"][-1] - lc["val"][-1]
    st.write("Gap between training and validation score: " + str(round(gap, 3)))
    if gap < 0.03:
        st.success("Small gap: the model is not overfitting.")
    elif gap < 0.08:
        st.warning("Moderate gap: slight overfitting.")
    else:
        st.error("Large gap: the model is overfitting.")
    st.caption("If the training score is much higher than the validation score, "
               "the model is memorising the data instead of learning real patterns.")
except FileNotFoundError:
    st.info("Run train.py again to enable the learning curve.")

st.header("Error analysis (where does the model go wrong?)")
try:
    err = joblib.load("error_analysis.joblib")
    err["Predicted"] = (err["Probability"] >= 0.5).astype(int)
    err["Group"] = "Correct"
    err.loc[(err["Predicted"] == 1) & (err["Actual"] == 0), "Group"] = "False alarm"
    err.loc[(err["Predicted"] == 0) & (err["Actual"] == 1), "Group"] = "Missed placed"
    n_fp = int((err["Group"] == "False alarm").sum())
    n_fn = int((err["Group"] == "Missed placed").sum())
    wrong = err[err["Group"] != "Correct"]
    st.write("Test students: " + str(len(err)) + " | Wrong predictions: " + str(len(wrong)))
    st.write("False alarms (predicted placed, actually not placed): " + str(n_fp))
    st.write("Missed placed students (predicted not placed, actually placed): " + str(n_fn))
    border = ((wrong["Probability"] - 0.5).abs() < 0.15).mean()
    st.write("Share of wrong predictions where the model was unsure (probability 35% to 65%): "
             + str(round(border * 100, 1)) + " %")

    err_num_cols = [c for c in number_cols if c in err.columns]
    st.write("Average value of each feature, group by group:")
    st.dataframe(err.groupby("Group")[err_num_cols].mean().round(2))

    st.write("The 10 most confident wrong predictions:")
    wrong = wrong.assign(Confidence=(wrong["Probability"] - 0.5).abs())
    st.dataframe(wrong.sort_values("Confidence", ascending=False).head(10).drop(columns=["Confidence"]))
    st.caption("Most mistakes happen for borderline students. The remaining ones are cases "
               "that the available features cannot explain, for example interview performance.")
except FileNotFoundError:
    st.info("Run train.py again to enable error analysis.")