import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.set_page_config(page_title="Apriori Algorithm", layout="centered")
st.title("🛒 Market Basket Analysis (Apriori)")

st.write("### Enter transactions (one cart per line)")
st.caption("Items must be separated by commas")

# -------------------------------
# USER INPUT
# -------------------------------
data = st.text_area(
    "Example:\nMilk,Bread,Butter\nBread,Jam\nMilk,Bread",
    height=180
)

# -------------------------------
# PARAMETERS
# -------------------------------
min_support = st.slider("Minimum Support", 0.1, 1.0, 0.3)
min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)

# -------------------------------
# PROCESS BUTTON
# -------------------------------
if st.button("Run Apriori Algorithm"):

    if not data.strip():
        st.error("❌ Please enter at least one transaction")
        st.stop()

    # -------------------------------
    # CREATE TRANSACTIONS
    # -------------------------------
    transactions = [
        [item.strip() for item in row.split(",") if item.strip()]
        for row in data.strip().split("\n")
    ]

    st.success("✅ Transactions created")
    st.write(transactions)

    # -------------------------------
    # ENCODING
    # -------------------------------
    te = TransactionEncoder()
    encoded_array = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(encoded_array, columns=te.columns_)

    st.write("### 🔐 Encoded Data")
    st.dataframe(encoded_df)

    # -------------------------------
    # APRIORI
    # -------------------------------
    frequent_itemsets = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        st.warning("⚠️ No frequent itemsets found")
        st.stop()

    frequent_itemsets["items"] = frequent_itemsets["itemsets"].apply(
        lambda x: ", ".join(x)
    )

    st.write("### 📊 Frequent Itemsets")
    st.dataframe(frequent_itemsets[["items", "support"]])

    # -------------------------------
    # ASSOCIATION RULES
    # -------------------------------
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        st.warning("⚠️ No association rules found")
        st.stop()

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(x))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(x))

    st.write("### 🔗 Association Rules")
    st.dataframe(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    )

    # -------------------------------
    # PLOT SUPPORT
    # -------------------------------
    st.write("### 📈 Frequent Itemsets Support Plot")
    fig, ax = plt.subplots()
    ax.bar(frequent_itemsets["items"], frequent_itemsets["support"], color="skyblue")
    ax.set_xlabel("Itemsets")
    ax.set_ylabel("Support")
    ax.set_title("Frequent Itemsets Support")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
z
