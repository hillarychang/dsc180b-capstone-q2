import pandas as pd
import numpy as np

# get balance and standard credit
def get_balance(acct, consumer, transactions):
    total_balance = acct.groupby("prism_consumer_id")["balance"].sum()
    consumer_balance = consumer.merge(
        pd.DataFrame(total_balance), on="prism_consumer_id", how="outer"
    )
    consumer_balance["std_credit"] = (
        consumer_balance["credit_score"] - consumer_balance["credit_score"].mean()
    ) / consumer_balance["credit_score"].std()

    return consumer_balance


def get_transaction_categories(transactions, categories):
    transaction_categories = transactions.merge(
        categories, how="left", left_on="category", right_on="category_id"
    )
    return transaction_categories


# get category occurences for DEBIT and CREDIT
def get_category_occurences_sums(transaction_categories, consumer_balance):
    outflow_occurences = (
        transaction_categories[transaction_categories["credit_or_debit"] == "DEBIT"]
        .groupby(["prism_consumer_id", "category_y"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # name columns for clarity
    outflow_occurences = outflow_occurences.rename(
        columns=lambda col: f"outflow_occurences_{col}"
        if col != "prism_consumer_id"
        else col
    )

    inflow_occurences = (
        transaction_categories[transaction_categories["credit_or_debit"] == "CREDIT"]
        .groupby(["prism_consumer_id", "category_y"])
        .size()  # Count number of unique occurrences
        .unstack(fill_value=0)  # Create one column per category_x
        .reset_index()
    )

    # name columns for clarity
    inflow_occurences = inflow_occurences.rename(
        columns=lambda col: f"inflow_occurrences_{col}"
        if col != "prism_consumer_id"
        else col
    )

    outflow_sums = (
        transaction_categories[transaction_categories["credit_or_debit"] == "DEBIT"]
        .groupby(["prism_consumer_id", "category_y"])["amount"]
        .sum()  # Count number of unique occurrences
        .unstack(fill_value=0)  # Create one column per category_x
        .reset_index()
    )
    # name columns for clarity
    outflow_sums = outflow_sums.rename(
        columns=lambda col: f"outflow_sums_{col}" if col != "prism_consumer_id" else col
    )

    inflow_sums = (
        transaction_categories[transaction_categories["credit_or_debit"] == "CREDIT"]
        .groupby(["prism_consumer_id", "category_y"])["amount"]
        .sum()  # Count number of unique occurrences
        .unstack(fill_value=0)  # Create one column per category_x
        .reset_index()
    )

    # name columns for clarity
    inflow_sums = inflow_sums.rename(
        columns=lambda col: f"inflow_sums_{col}" if col != "prism_consumer_id" else col
    )

    features = (
        outflow_occurences.merge(outflow_sums, how="left", on="prism_consumer_id")
        .merge(inflow_occurences, how="left", on="prism_consumer_id")
        .merge(inflow_sums, how="left", on="prism_consumer_id")
    )

    consumer_features = consumer_balance.merge(
        features, how="left", on="prism_consumer_id"
    )

    return consumer_features


def one_hot_accounts(acct, consumer_features):
    df = acct.copy()
    df["balance_data"] = pd.to_datetime(df["balance_date"])
    one_hot = pd.get_dummies(df["account_type"], prefix="account_type")
    one_hot_aggregated = (
        pd.concat([df[["prism_consumer_id"]], one_hot], axis=1)
        .groupby("prism_consumer_id")
        .sum()
    )

    all_features = consumer_features.merge(one_hot_aggregated, on="prism_consumer_id")
    return all_features

def all_cat_percent(all_features, transactions, consumer, categories):
    def get_cat_percent(df, category="all"):
        user_cat = (
            df.groupby(["prism_consumer_id", "category"]).amount.sum().reset_index()
        )
        pivot_df = user_cat.pivot(
            index="prism_consumer_id", columns="category", values="amount"
        )
        pivot_df.fillna(0, inplace=True)
        pivot_df["total"] = pivot_df.sum(axis=1)
        for i in user_cat.category.unique():
            pivot_df[i] = (pivot_df[i] / pivot_df["total"]) * 100
            pivot_df = pivot_df.rename(columns={i: categories.loc[i].category})
        category_percent = consumer.merge(
            pivot_df, on="prism_consumer_id", how="outer"
        ).drop(columns=["evaluation_date", "credit_score"])
        category_percent = category_percent.drop(columns=["DQ_TARGET"])
        category_percent = category_percent.set_index("prism_consumer_id")
        column_names = []
        for i in category_percent.columns:
            column_names.append(f"{category}_percentage_{i}")
        category_percent.columns = column_names
        return category_percent

    trxn = transactions.copy()
    credit_trxn = trxn[trxn["credit_or_debit"] == "CREDIT"]
    debit_trxn = trxn[trxn["credit_or_debit"] == "DEBIT"]

    credit_cat_features = get_cat_percent(credit_trxn, "credit")
    debit_cat_features = get_cat_percent(debit_trxn, "debit")
    trxn_cat_features = get_cat_percent(trxn, "trxn")

    all_cat_features = credit_cat_features.merge(
        debit_cat_features, left_index=True, right_index=True
    )
    all_cat_features = all_cat_features.merge(
        trxn_cat_features, left_index=True, right_index=True
    )

    all_features = all_features.merge(all_cat_features, on = "prism_consumer_id")
    return all_features

def running_total(all_features, transactions):
    def running_total_statistics(df, time_frame="1D"):
        df["posted_date"] = pd.to_datetime(df["posted_date"])
        df.loc[df["credit_or_debit"] == "DEBIT", "amount"] *= -1
        df = (
            df.groupby(
                ["prism_consumer_id", pd.Grouper(key="posted_date", freq=time_frame)]
            )
            .amount.sum()
            .reset_index()
        )
        df = df.sort_values("posted_date")
        df["running_total"] = df.groupby("prism_consumer_id")["amount"].cumsum()
        new_df = df.groupby("prism_consumer_id").agg(
            {"running_total": ["mean", "median", "var"]}
        )
        new_df.columns = [
            f"{time_frame}_mean",
            f"{time_frame}_median",
            f"{time_frame}_var",
        ]
        return new_df

    month = running_total_statistics(transactions.copy(), time_frame="1M")
    week = running_total_statistics(transactions.copy(), time_frame="1W")
    day = running_total_statistics(transactions.copy(), time_frame="1D")
    time_features = month.merge(week, left_index=True, right_index=True)
    time_features = time_features.merge(day, left_index=True, right_index=True)
    all_features = all_features.merge(time_features, on = "prism_consumer_id")
    return all_features

