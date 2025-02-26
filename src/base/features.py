import pandas as pd
import numpy as np
import polars as pl


def get_datasets():
    categories = pd.read_csv("../../data_q2/q2-ucsd-cat-map.csv")
    consumer = pd.read_parquet("../../data_q2/q2-ucsd-consDF.pqt")
    acct = pd.read_parquet("../../data_q2/q2-ucsd-acctIDF.pqt")
    transactions = pd.read_parquet("../../data_q2/q2-ucsd-trxnDF.pqt")
    transactions["amount"] = transactions["amount"].where(
        transactions["credit_or_debit"] == "DEBIT", -transactions["amount"]
    )
    return categories, consumer, acct, transactions


def get_transaction_categories(transactions, categories):
    transaction_categories = transactions.merge(
        categories, how="left", left_on="category", right_on="category_id"
    )
    transaction_categories = transaction_categories.drop(columns=["category_x"])
    transaction_categories.rename(columns={"category_y": "category"}, inplace=True)
    transaction_categories = transaction_categories[
        ~transaction_categories["category"].isin(
            [
                "UNEMPLOYMENT_BENEFITS",
                "EDUCATION",
                "HOME_IMPROVEMENT",
                "HEALTHCARE_MEDICAL",
                "CHILD_DEPENDENTS",
                "PENSION",
            ]
        )
    ]

    return transaction_categories


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

def count_account_types(acct, consumer_features):
    # Group by consumer ID and account type, count occurrences
    account_counts = (
        acct.groupby(["prism_consumer_id", "account_type"])
        .size()
        .unstack(fill_value=0)
        .add_prefix("num_accounts_")  # Prefix for clarity
    )

    # Merge with consumer features
    all_features = consumer_features.merge(
        account_counts, on="prism_consumer_id", how="left"
    ).fillna(0)  # Fill NaN with 0 for consumers with no accounts of certain types

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

    all_features = all_features.merge(all_cat_features, on="prism_consumer_id")
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
    all_features = all_features.merge(time_features, on="prism_consumer_id")
    return all_features


def get_categorical_features(all_features, transaction_categories, acct):
    """
    Extracts categorical features from the provided data in a more efficient manner.

    Args:
        data (dict): A dictionary containing DataFrames.

    Returns:
        DataFrame: DataFrame containing categorical features.
    """
    # Convert 'posted_date' to datetime and create a 'month' column.
    transaction_categories["datetime"] = pd.to_datetime(
        transaction_categories["posted_date"]
    )
    transaction_categories["month"] = transaction_categories["datetime"].dt.strftime(
        "%Y-%m"
    )

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

    by_category = (
        transaction_categories[["prism_consumer_id", "category", "month", "amount"]]
        .groupby(["prism_consumer_id", "category", "month"])
        .sum()
        .reset_index()
    )

    by_category = by_category.merge(
        consumer_category_months_df,
        on=["prism_consumer_id", "category", "month"],
        how="right",
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

    def polars_categorical_features(transaction_categories):
        df = pl.DataFrame(transaction_categories)

        advanced_metrics = df.group_by(["prism_consumer_id", "category"]).agg(
            [
                pl.col("amount").median().alias("median"),
                pl.col("amount").count().alias("count"),
                pl.col("amount").skew().alias("skewness"),
                (
                    pl.col("amount").quantile(0.75) - pl.col("amount").quantile(0.25)
                ).alias("iqr"),
                (pl.col("amount").std() / pl.col("amount").mean())
                .fill_nan(0)
                .alias("coef_variation"),
            ]
        )

        return advanced_metrics

    advanced_metrics = polars_categorical_features(transaction_categories).to_pandas()
    pivot_advanced = advanced_metrics.pivot_table(
        index="prism_consumer_id",
        columns="category",
        values=["median", "count", "skewness", "iqr", "coef_variation"],
    )
    pivot_advanced.columns = [f"{col[1]}_{col[0]}" for col in pivot_advanced.columns]
    pivot_advanced = pivot_advanced.reset_index()
    pivot_advanced = pivot_advanced.fillna(0)
    all_features = all_features.merge(
        pivot_advanced, on="prism_consumer_id", how="left"
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
    pivot_df = pivot_df.merge(acct_on_cons, on="prism_consumer_id", how="left")

    all_features = all_features.merge(pivot_df, on="prism_consumer_id", how="left")

    return all_features


def get_categorical_features2(all_features, transaction_categories, acct):
    # Convert 'posted_date' to datetime and create 'month' column
    transaction_categories["datetime"] = pd.to_datetime(
        transaction_categories["posted_date"]
    )
    transaction_categories["month"] = transaction_categories["datetime"].dt.strftime(
        "%Y-%m"
    )

    # Generate all consumer-category-month combinations using vectorized operations
    consumer_intervals = (
        transaction_categories.groupby("prism_consumer_id")["datetime"]
        .agg(min_date="min", max_date="max")
        .reset_index()
    )

    consumer_intervals["min_month"] = (
        consumer_intervals["min_date"].dt.to_period("M").dt.start_time
    )
    consumer_intervals["max_month"] = (
        consumer_intervals["max_date"].dt.to_period("M").dt.start_time
    )

    # Generate all months for each consumer
    consumer_intervals["months"] = consumer_intervals.apply(
        lambda row: pd.date_range(
            start=row["min_month"], end=row["max_month"], freq="MS"
        ),
        axis=1,
    )
    consumer_months = consumer_intervals.explode("months")[
        ["prism_consumer_id", "months"]
    ]
    consumer_months["month"] = consumer_months["months"].dt.strftime("%Y-%m")
    consumer_months = consumer_months.drop(columns="months")

    # Cross join with categories
    categories = transaction_categories["category"].unique()
    consumer_category_months_df = pd.merge(
        consumer_months.assign(key=1),
        pd.DataFrame({"category": categories, "key": 1}),
        on="key",
    ).drop(columns="key")

    # Aggregate amounts by consumer-category-month and merge to fill missing with 0
    by_category = (
        transaction_categories.groupby(["prism_consumer_id", "category", "month"])[
            "amount"
        ]
        .sum()
        .reset_index()
    )

    by_category = pd.merge(
        consumer_category_months_df,
        by_category,
        on=["prism_consumer_id", "category", "month"],
        how="left",
    ).fillna({"amount": 0})

    # Calculate diffs between consecutive months
    by_category = by_category.sort_values(["prism_consumer_id", "category", "month"])
    by_category["diffs"] = (
        by_category.groupby(["prism_consumer_id", "category"])["amount"]
        .diff()
        .fillna(0)
    )

    # Aggregate mean and std of diffs
    metrics = (
        by_category.groupby(["prism_consumer_id", "category"])["diffs"]
        .agg(["mean", "std"])
        .fillna(0)
        .reset_index()
    )

    # Pivot metrics
    pivot_df = metrics.pivot_table(
        index="prism_consumer_id",
        columns="category",
        values=["mean", "std"],
        fill_value=0,
    )
    # Correct column naming
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Advanced metrics using Pandas
    advanced_metrics = (
        transaction_categories.groupby(["prism_consumer_id", "category"])
        .agg(
            median=("amount", "median"),
            count=("amount", "count"),
            skewness=("amount", "skew"),
            q1=("amount", lambda x: x.quantile(0.25)),
            q3=("amount", lambda x: x.quantile(0.75)),
            mean_amount=("amount", "mean"),
            std_amount=("amount", "std"),
        )
        .reset_index()
    )

    advanced_metrics["iqr"] = advanced_metrics["q3"] - advanced_metrics["q1"]
    advanced_metrics["coef_variation"] = (
        advanced_metrics["std_amount"] / advanced_metrics["mean_amount"]
    ).fillna(0)
    advanced_metrics.drop(
        columns=["q1", "q3", "mean_amount", "std_amount"], inplace=True
    )

    # Pivot advanced metrics
    pivot_advanced = advanced_metrics.pivot_table(
        index="prism_consumer_id",
        columns="category",
        values=["median", "count", "skewness", "iqr", "coef_variation"],
        fill_value=0,
    )
    pivot_advanced.columns = [f"{col[1]}_{col[0]}" for col in pivot_advanced.columns]
    pivot_advanced = pivot_advanced.reset_index()

    # Merge advanced metrics into all_features
    all_features = all_features.merge(
        pivot_advanced, on="prism_consumer_id", how="left"
    )

    # Merge account balances
    acct_on_cons = acct.groupby("prism_consumer_id")["balance"].sum().reset_index()
    pivot_df = pivot_df.merge(acct_on_cons, on="prism_consumer_id", how="left").fillna(
        0
    )

    # Merge metrics into all_features
    all_features = all_features.merge(
        pivot_df, on="prism_consumer_id", how="left"
    ).fillna(0)

    return all_features

def get_total_transactions(all_features, transaction_categories, transactions_needed = 25):
    transaction_totals = transaction_categories.groupby("prism_consumer_id")[
        "category"
    ].count()

    all_features = all_features.merge(transaction_totals, on = 'prism_consumer_id', how = 'left')
    all_features = all_features.rename(columns={"category": "transactions"})
    all_features['transactions'] = all_features['transactions'].fillna(0)
    # all_features = all_features[all_features["transactions"] > transactions_needed]

    return all_features


def get_income(all_features, df, tolerance=5):
    def calculate_time_diff(dates):
        dates = pd.to_datetime(dates)
        sorted_dates = dates.sort_values()
        time_diff = sorted_dates.diff().dropna()
        return time_diff

    def is_regular(time_diffs):
        week_range = (7 - tolerance, 7 + tolerance)
        bi_week_range = (14 - tolerance, 14 + tolerance)
        month_range = (30 - tolerance, 30 + tolerance)

        regular_intervals = time_diffs.apply(lambda x: 
            (week_range[0] <= x.days <= week_range[1]) or
            (bi_week_range[0] <= x.days <= bi_week_range[1]) or
            (month_range[0] <= x.days <= month_range[1])
        )
        
        return regular_intervals.all()

    regular_transactions = []
    df['posted_date'] = pd.to_datetime(df['posted_date'])

    for (user, category), group in df.groupby(['prism_consumer_id', 'category']):
        date_list = group['posted_date']
        time_diffs = calculate_time_diff(date_list)

        if len(time_diffs) < 2:
            regular_transactions.append((user, category, False))
            continue

        is_reg = is_regular(time_diffs)
        regular_transactions.append((user, category, is_reg))
    
    reg_df = pd.DataFrame(regular_transactions, columns=['prism_consumer_id', 'category', 'is_regular'])

    df = df.merge(reg_df, on=['prism_consumer_id', 'category'], how='left')

    df['is_income'] = (df['is_regular'] & (df['credit_or_debit'] == 'CREDIT'))

    days_df = df.groupby('prism_consumer_id').agg(first_posted_date=('posted_date', 'min'),
                                                   last_posted_date=('posted_date', 'max')).reset_index()
    days_df['days'] = (pd.to_datetime(days_df['last_posted_date']) - pd.to_datetime(days_df['first_posted_date'])).dt.days

    

    income_df = df[df['is_income']].groupby('prism_consumer_id')['amount'].sum().reset_index()
    income_df = income_df.rename(columns={'amount': 'income'})

    income_df = income_df.merge(days_df[['prism_consumer_id', 'days']], on='prism_consumer_id', how='left')
    income_df['income_per_day'] = income_df['income'] / income_df['days']
    all_features = all_features.merge(income_df[['prism_consumer_id', 'income_per_day']], on ='prism_consumer_id')
    all_features['income_per_day'] *= -1
    return all_features
