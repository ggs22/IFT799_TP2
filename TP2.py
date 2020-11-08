import seaborn as sns
import numpy as np
import pandas as pd
import sklearn as sk


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

    if age_brackets[0][0] > age or age_brackets[len(age_brackets)-1][0] <= age:
        raise ValueError('The age entered does not belong in any considered age bracket!')

    for i in range(0, len(age_brackets)-1):
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

    for i in range(0, len(income_brackets)-1):
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


# load data
bank_data = pd.read_csv('bank-data.csv')

# preliminary data examination
print_section(bank_data.nunique().to_string())

# select all numerical columns
numeric_cols = bank_data.select_dtypes(include=['float64', 'int64'])

# get min and max values of all numeric columns
print_section(np.max(numeric_cols).to_string(), title="Max values")
print_section(np.min(numeric_cols).to_string(), title="Min values")



input('Press any key to quit...')