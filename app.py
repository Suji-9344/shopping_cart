import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori Algorithm App", layout="wide")

st.title("🛒 Market Basket Analysis – Apriori Algorithm")

# ===============================
# Upload Dataset
# ===============================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload shopping_cart.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ===============================
    # Create Instances
    # ===============================
    st.subheader("🧩 Transaction Instances")

    transactions = []
    for i in range(len(df)):
        transactions.append(df.iloc[i, 0].split(","))

    st.write("Sample Transaction:")
    st.write(transactions[0])

    # ===============================
    # Encoding
    # ===============================
    st.subheader("🔐 Encoding (One-Hot Encoding)")

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(te_array, columns=te.columns_)

    st.dataframe(encoded_df.head())

    # ===============================
    # Apriori Parameters
    # ===============================
    st.sidebar.header("⚙️ Apriori Parameters")

    min_support = st.sidebar.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=1.0,
        value=0.2,
        step=0.01
    )

    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.01,
        max_value=1.0,
        value=0.6,
        step=0.01
    )

    # ===============================
    # Apply Apriori
    # ===============================
    frequent_items = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True
    )

    frequent_items["items"] = frequent_items["itemsets"].apply(
        lambda x: ", ".join(list(x))
    )

    st.subheader("📊 Frequent Itemsets")
    st.dataframe(frequent_items[["items", "support"]])

    # ===============================
    # Association Rules
    # ===============================
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=min_confidence
    )

    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(list(x))
        )
        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(list(x))
        )

        st.subheader("🔗 Association Rules")
        st.dataframe(
            rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        )

        # ===============================
        # Plot Diagram
        # ===============================
        st.subheader("📈 Support Plot")

        fig = plt.figure()
        plt.bar(frequent_items["items"], frequent_items["support"])
        plt.xlabel("Itemsets")
        plt.ylabel("Support")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.warning("⚠️ No rules found. Try lowering confidence/support.")

else:
    st.info("⬅️ Upload a shopping_cart.csv file to begin")
