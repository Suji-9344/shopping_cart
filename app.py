import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori Algorithm App", layout="wide")

st.title("🛒 Market Basket Analysis – Apriori Algorithm")

# ===============================
# USER INPUT
# ===============================
st.subheader("✍️ Enter Transactions")

st.info("Enter one shopping cart per line (items separated by commas)")

user_input = st.text_area(
    "Example:\nMilk,Bread,Butter\nBread,Jam\nMilk,Bread",
    height=200
)

if user_input:
    # ===============================
    # Create INSTANCES
    # ===============================
    transactions = []

    for line in user_input.strip().split("\n"):
        transactions.append([item.strip() for item in line.split(",")])

    st.subheader("🧩 Transaction Instances")
    st.write(transactions)

    # ===============================
    # Encoding
    # ===============================
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(te_array, columns=te.columns_)

    st.subheader("🔐 Encoded Data")
    st.dataframe(encoded_df)

    # ===============================
    # Apriori Parameters
    # ===============================
    st.sidebar.header("⚙️ Apriori Parameters")

    min_support = st.sidebar.slider(
        "Minimum Support",
        0.01, 1.0, 0.3, 0.01
    )

    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        0.01, 1.0, 0.6, 0.01
    )

    # ===============================
    # Apply Apriori
    # ===============================
    frequent_items = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True
    )

    if not frequent_items.empty:
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

        if not rules.emp
