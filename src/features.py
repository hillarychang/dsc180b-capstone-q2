def build_features():
    pass
    
def add_token(data):
    df = data.copy()

    def add_text(memo, added_tokens):
        memo += ''.join(added_tokens)
        return memo

    def whole_dollar_amount(amount):
        if amount % 1 == 0:
            return ' <W_D>'
        return ''

    def day(date):
        return f" <D_{date.day}>"

    def month(date):
        return f" <M_{date.month}>"

    df['memo'] = df.apply(
        lambda row: add_text(row['memo'], 
                             [whole_dollar_amount(row['amount']), 
                              day(row['posted_date']),
                             month(row['posted_date'])]), axis=1
    )

    return df
