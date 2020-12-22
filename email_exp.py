import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('/Users/apple4u/Downloads/email_3days.csv')
    df = df.drop(['id', 'date', 'content', 'cc_field', 'bcc_field', 'pc', 'size', 'attachments'], axis=1)
    print(df.head())
    users_unique = df.user.unique()
    print(users_unique)
    users_all = df.user.values
    # df = df.drop(['AGB0643', 'ABC0253', 'AGB0643', 'AIP0982', 'AJR0932', 'AKK0057', 'ALD0544',
    #               'AMH0794', 'AMR0400', 'ARB0881', 'ASG0692', 'ASS0517', 'BAM0636', 'BBS0422',
               #   'BJH0811', 'BSS0463', 'BTL0722', 'CAB0614', ])


    print(len(users_all))

    dict = {}
    df_new = pd.DataFrame(columns=['usser', 'too'])
    for index, row in df.iterrows():
        to_field = str(row['to_field']).split(";")  # Ainsley.MacKensie.Doyle@dtaa.com etc.
        for i in range(len(to_field)):
            if to_field[i].__contains__('@dtaa.com'):  # filter out non dtaa emails
                #to_field[i] = to_field[i].split(".")
                #to_new = ''
                #for j in range(len(to_field[i])):
                #    to_new = to_new + to_field[i][j][0]
                #to_new = to_new[:-1]
                df_new = df_new.append({'useer': row['user'],
                                        'froom': row['from_field'],
                                        'too': to_field[i]}, ignore_index=True)

print(df_new.head())
