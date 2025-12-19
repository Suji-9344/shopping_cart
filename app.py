import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Apriori Max Confidence", layout="centered")
st.title("🛒 Apriori Algorithm - Max Confidence Rule")

st.write("Enter transactions (one cart per line). Items separated by commas.")

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

# -------------------------------
# PROCESS BUTTON
# -------------------------------
if st.button("Run Apriori"):

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

    # -------------------------------
    # ENCODING
    # -------------------------------
    te = TransactionEncoder()
    encoded_array = te.fit(transactions).transform(transactions)
    encoded_df = pd.DataFrame(encoded_array, columns=te.columns_)

    # -------------------------------
    # APRIORI
    # -------------------------------
    frequent_itemsets = apriori(
        encoded_df,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        st.warning("⚠️ No frequent itemsets found for this support level")
        st.stop()

    # -------------------------------
    # ASSOCIATION RULES
    # -------------------------------
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0)
    if rules.empty:
        st.warning("⚠️ No association rules found")
        st.stop()

    # Convert sets to string for display
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(x))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(x))

    # Find rule(s) with maximum confidence
    max_conf = rules["confidence"].max()
    max_rules = rules[rules["confidence"] == max_conf]

    st.subheader("🔝 Rule(s) with Maximum Confidence")
    st.dataframe(max_rules[["antecedents", "consequents", "support", "confidence", "lift"]])
