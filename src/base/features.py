import pandas as pd
import numpy as np

def get_datasets():
    categories = pd.read_csv("../../data_q2/q2-ucsd-cat-map.csv")
    consumer = pd.read_parquet("../../data_q2/q2-ucsd-consDF.pqt")
    acct = pd.read_parquet("../../data_q2/q2-ucsd-acctIDF.pqt")
    transactions = pd.read_parquet("../../data_q2/q2-ucsd-trxnDF.pqt")
    transactions["amount"] = transactions["amount"].where(
        transactions["credit_or_debit"] == "DEBIT", -transactions["amount"]
    )
    return categories, consumer, acct, transactions

# get balance and standard credit
def get_balance(acct, consumer):
    total_balance = acct.groupby("prism_consumer_id")["balance"].sum()
    consumer_balance = consumer.merge(
        pd.DataFrame(total_balance), on="prism_consumer_id", how="outer"
    )

    return consumer_balance


def get_transaction_categories(transactions, categories):
    transaction_categories = transactions.merge(
        categories, how="left", left_on="category", right_on="category_id"
    )
    transaction_categories = transaction_categories.drop(columns = ['category_x'])
    transaction_categories.rename(columns={"category_y": "category"})
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

def get_categorical_features(all_features, transaction_categories):
    """
    Extracts categorical features from the provided data in a more efficient manner.

    Args:
        data (dict): A dictionary containing DataFrames.

    Returns:
        DataFrame: DataFrame containing categorical features.
    """
    # Convert 'posted_date' to datetime and create a 'month' column.
    transaction_categories["datetime"] = pd.to_datetime(transaction_categories["posted_date"])
    transaction_categories["month"] = transaction_categories["datetime"].dt.strftime("%Y-%m")

    # Generate the permutations of consumers, categories, and months
    consumer_intervals = (
        transaction_categories[["prism_consumer_id", "month"]]
        .groupby("prism_consumer_id")
        .agg(["min", "max"])
    )
    consumer_intervals.columns = ["min", "max"]
    consumer_intervals = consumer_intervals.to_dict()
    categories = sorted(transaction_categories["category"].unique())

    consumer_category_months = []
    for con in consumer_intervals["min"].keys():
        consumer_min = consumer_intervals["min"][con]
        consumer_max = consumer_intervals["max"][con]
        month_range = pd.date_range(consumer_min, consumer_max, freq="1M")
        month_range = [d.strftime("%Y-%m") for d in month_range] + [consumer_max]

        for category in categories:
            for month in month_range:
                consumer_category_months.append(
                    {
                        "prism_consumer_id": con,
                        "month": month,
                        "category": category,
                    }
                )

    consumer_category_months_df = pd.DataFrame(consumer_category_months)

    # Group the transaction_categories data by 'prism_consumer_id', 'category', and 'month' and aggregate 'amount'
    by_category = (
        transaction_categories[["prism_consumer_id", "category", "month", "amount"]]
        .groupby(["prism_consumer_id", "category", "month"])
        .sum()
        .reset_index()
    )

    # Merge the generated consumer-category-month combinations with the aggregated data
    by_category = by_category.merge(
        consumer_category_months_df,
        on=["prism_consumer_id", "category", "month"],
        how="right",
    )

    # Fill any missing values with 0
    by_category = by_category.fillna(0)

    # Calculate the difference of 'amount' for each consumer-category group (diff of consecutive months)
    by_category["diffs"] = by_category.groupby(["prism_consumer_id", "category"])[
        "amount"
    ].transform(lambda x: x.diff())

    # Aggregate the mean and std of the amounts for each consumer-category group
    metrics = (
        by_category.drop(columns="month")
        .groupby(["prism_consumer_id", "category"])
        .agg(["mean", "std"])
    )

    # Aggregate account balances by 'prism_consumer_id'
    acct_on_cons = (
        acct[["prism_consumer_id", "balance"]].groupby("prism_consumer_id").sum()
    )

    # Create a pivot table for the consumer-category statistics
    pivot_df = metrics.pivot_table(index="prism_consumer_id", columns="category")
    pivot_df.columns = [f"{col[2]}_{col[0]}_{col[1]}" for col in pivot_df.columns]

    # Fill any NaN values with 0
    pivot_df = pivot_df.fillna(0).reset_index()

    # Merge the consumer statistics and account balance data
    pivot_df = pivot_df.merge(consumer, on="prism_consumer_id", how="left")
    pivot_df = pivot_df.merge(acct_on_cons, on="prism_consumer_id", how="left")

    return pivot_df