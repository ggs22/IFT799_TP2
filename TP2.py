# IFT799 - TP2
# 9 novembre 2020
# Gabriel  Gibeau Sanchez - gibg2501

import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


def get_age_bracket(age):
    # see https://www.statcan.gc.ca/eng/concepts/definitions/age1
    age_brackets = ((15, '13'),
                    (20, '14'),
                    (25, '15'),
                    (30, '16'),
                    (35, '17'),
                    (40, '18'),
                    (45, '19'),
                    (50, '20'),
                    (55, '21'),
                    (60, '22'),
                    (65, '23'),
                    (70, 'NA'))

    if age_brackets[0][0] > age or age_brackets[len(age_brackets) - 1][0] <= age:
        raise ValueError('The age entered does not belong in any considered age bracket!')

    for i in range(0, len(age_brackets) - 1):
        if age_brackets[i][0] <= age < age_brackets[i + 1][0]:
            return age_brackets[i][1]


def get_income_bracket(income):
    income_brackets = ((5000, 'A'),
                       (10000, 'B'),
                       (15000, 'C'),
                       (20000, 'D'),
                       (25000, 'E'),
                       (30000, 'F'),
                       (35000, 'G'),
                       (40000, 'H'),
                       (45000, 'I'),
                       (50000, 'J'),
                       (55000, 'K'),
                       (60000, 'L'),
                       (65000, 'M'))

    if income_brackets[0][0] > income or income_brackets[len(income_brackets) - 1][0] <= income:
        raise ValueError('The income entered does not belong in any considered income bracket!')

    for i in range(0, len(income_brackets) - 1):
        if income_brackets[i][0] <= income < income_brackets[i + 1][0]:
            return income_brackets[i][1]


def print_section(text, title=None):
    sep = "======================"
    print("\n")

    if title is not None:
        print(title)
    print(sep)
    print(text)
    print(sep)


# Load data
bank_data = pd.read_csv('bank-data.csv')

print_section(bank_data.nunique().to_string())

# Select all numerical columns
numeric_cols = bank_data.select_dtypes(include=['float64', 'int64'])

# Get min and max values of all numeric columns
print_section(np.max(numeric_cols).to_string(), title="Max values")
print_section(np.min(numeric_cols).to_string(), title="Min values")

# Convert numerical values to categories
preprocessed_data = bank_data.copy()
preprocessed_data.loc[:, 'age'] = bank_data['age'].apply(get_age_bracket)
preprocessed_data.loc[:, 'income'] = bank_data['income'].apply(get_income_bracket)
preprocessed_data.loc[:, 'children'] = bank_data['children'].astype('str')

# Sanity check
print_section(preprocessed_data.nunique().to_string())

records = list()
tupple = list()
for index, row in preprocessed_data.iterrows():
    tupple.clear()
    for i in range(1, len(row)):
        if i >= 5 and i != 6:
            if row[i] == 'YES':
                tupple.append(preprocessed_data.keys()[i])
        elif i == 6:
            tupple.append(f'{row[i]}-{preprocessed_data.keys()[i]}')
        else:
            tupple.append(row[i])
    records.append(tupple.copy())

# Display a sample of the pre-processed data
print('\n')
for i in range(0, 5):
    print(records[i])

# Apriori processing
te = TransactionEncoder()
te_ary = te.fit(records).transform(records)
frequent_itemsets = apriori(pd.DataFrame(te_ary, columns=te.columns_), min_support=0.1, use_colnames=True)
frequent_itemsets['item set size'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets.sort_values(by=['support'], inplace=True, ascending=False)

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Filter out resulst to keep only consequents containing 'pep' and 'mortgage'
for index, row in rules.iterrows():
    if not ('pep' in row['consequents'] or 'mortgage' in row['consequents']):
        rules.drop(index, axis=0, inplace=True)

rules_by_conf = rules.sort_values(by=['confidence'], ascending=False)
rules_by_lift = rules.sort_values(by=['lift'], ascending=False)

# Create a custom decision criteria to select rules
rules['average lift and confidence'] = np.average([rules.loc[:, 'confidence'], rules.loc[:, 'lift']], axis=0)
rules.sort_values(by=['average lift and confidence'], inplace=True, ascending=False)

print_section(rules_by_conf.iloc[0:10, :].to_string(), title='By confidence')
print_section(rules_by_lift.iloc[0:10, :].to_string(), title='By lift')
print_section(rules.iloc[0:10, [0, 1, 5, 6, 9]].to_string(), title='By custom criteria')

potential_pep_customers = list()
for _, row in preprocessed_data.iterrows():
    if row['children'] == '1' and row['pep'] == 'NO':
        potential_pep_customers.append(row['id'])

print(f'\nThere are {len(potential_pep_customers)} potential PEP customer:'
      f'{potential_pep_customers[0:int(len(potential_pep_customers)/2)]}\n'
      f'{potential_pep_customers[int(len(potential_pep_customers)/2):len(potential_pep_customers)]}')

input('Press any key to quit...')
