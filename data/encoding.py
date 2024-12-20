import pandas as pd

demographics = pd.read_csv("demographics.csv")
print(demographics.head(20))
demographics['highest_education'] = demographics['highest_education'].str.replace(' ', '_').str.lower()
demographics['region'] = demographics['region'].str.replace(' ', '_').str.lower()
demographics['final_result'] = demographics['final_result'].str.lower()
demographics['disability'] = demographics['disability'].str.lower()
demographics['imd_band'] = demographics['imd_band'].str.replace('%', '').str.lower()
demographics['age_band'] = demographics['age_band'].str.replace('55<=', 'over55').str.lower()
demographics['gender'] = demographics['gender'].str.lower()
dummies = pd.get_dummies(demographics, columns=['gender',
                                                'disability',
                                                'highest_education',
                                                'region',
                                                'imd_band',
                                                'age_band',
                                                'studied_credits',
                                                'final_result',
                                                'num_of_prev_attempts'])
print(dummies.head(20))


# Put studied_credits into groups
bins = [0, 29, 59, 89, 119, 149, float('inf')]
labels = ['0-29', '30-59', '60-89', '90-119', '120-149', '150over']

demographics['credits_group'] = pd.cut(demographics['studied_credits'], bins=bins, labels=labels, right=True)
credits_group_one_hot = pd.get_dummies(demographics['credits_group'], prefix='group_studied_credits')
demographics = pd.concat([demographics, dummies, credits_group_one_hot], axis=1)

print(demographics.head(20))

demographics.to_csv("demographics_encoded.csv")