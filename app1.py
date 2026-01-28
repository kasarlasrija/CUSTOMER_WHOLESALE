import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- 1️⃣ App Title & Description ----------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

# ---------------- Load Dataset ----------------
DATA_PATH = r"C:\Users\kasar\Downloads\Wholesale customers data.csv"

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- 2️⃣ Input Section (Sidebar) ----------------
st.sidebar.header("⚙️ Clustering Controls")

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

feature_1 = st.sidebar.selectbox(
    "Select Feature 1", numeric_columns, index=0
)
feature_2 = st.sidebar.selectbox(
    "Select Feature 2", numeric_columns, index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)", min_value=2, max_value=10, value=3
)

random_state = st.sidebar.number_input(
    "Random State (Optional)", value=42, step=1
)

# ---------------- 3️⃣ Clustering Control ----------------
run = st.sidebar.button("🟦 Run Clustering")

if run:
    # ---------------- Data Preparation ----------------
    X = df[[feature_1, feature_2]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- K-Means Model ----------------
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        init="k-means++"
    )

    df["Cluster"] = kmeans.fit_predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # ---------------- 4️⃣ Visualization Section ----------------
    st.subheader("📊 Customer Clusters Visualization")

    fig, ax = plt.subplots()

    ax.scatter(
        df[feature_1],
        df[feature_2],
        c=df["Cluster"]
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=300,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Segmentation using K-Means")
    ax.legend()

    st.pyplot(fig)

    # ---------------- 5️⃣ Cluster Summary Section ----------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df.groupby("Cluster")
        .agg(
            Count=("Cluster", "count"),
            Avg_Feature_1=(feature_1, "mean"),
            Avg_Feature_2=(feature_2, "mean")
        )
        .reset_index()
    )

    st.dataframe(summary)

    # ---------------- 6️⃣ Business Interpretation Section ----------------
    st.subheader("💡 Business Interpretation")

    for _, row in summary.iterrows():
        st.write(
            f"🟢 **Cluster {int(row['Cluster'])}**: "
            f"Customers show similar spending levels in "
            f"{feature_1} and {feature_2}. "
            f"This group can be targeted with customized marketing strategies."
        )

    # ---------------- 7️⃣ User Guidance / Insight Box ----------------
    st.info(
        "📌 Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )
