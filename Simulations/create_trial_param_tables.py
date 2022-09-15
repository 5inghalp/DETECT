import pandas as pd
import numpy as np

fixed_parameters = {'Min_Signpost_Codes': 50,
                    'Max_Signpost_Codes': 200,
                    'Min_Noise_Codes': 5,
                    'Max_Noise_Codes': 300}

to_concat = []

defaults = {'Total_Codes': 1500,
            'Num_People': 150000,
            'Num_Outcomes': 10,
            'Max_Outcome_Prev': 0.1,
            'Co_Occurrence_Mean': 0.04,
            'Max_Noise_Group_Prev': 0.5,
            'Num_Noise_Groups': 20,
            'Co_Occurrence_Noise_Mean': 0.01,
            'Random_Noise_Scale': 0.05}

# Maximum Outcome Prevalence Tests
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'max_outcome_prev'
new_df = pd.concat([new_row] * 4, axis=1).transpose()
new_df['Max_Outcome_Prev'] = [0.005, 0.01, 0.05, 0.1]

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Max_Outcome_Prev'))

# Co-Occurrence Mean Tests
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'co_occur_mean'
new_df = pd.concat([new_row] * 10, axis=1).transpose()
new_df['Co_Occurrence_Mean'] = np.arange(0.01, 0.11, 0.01)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Co_Occurrence_Mean'))

# Noise Co-Occurrence Mean Tests
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'co_occur_noise_mean'
new_df = pd.concat([new_row] * 10, axis=1).transpose()
new_df['Co_Occurrence_Noise_Mean'] = np.arange(0.01, 0.11, 0.01)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Co_Occurrence_Noise_Mean'))

# Max Noise Group Prevalence
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'noise_group_prev'
new_df = pd.concat([new_row] * 10, axis=1).transpose()
new_df['Max_Noise_Group_Prev'] = np.arange(0.1, 1.1, 0.1)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Max_Noise_Group_Prev'))

# Number of noise groups
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'num_noise_groups'
new_df = pd.concat([new_row] * 5, axis=1).transpose()
new_df['Num_Noise_Groups'] = np.arange(10, 60, 10)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Num_Noise_Groups'))

# Random noise scale
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'random_scale'
new_df = pd.concat([new_row] * 5, axis=1).transpose()
new_df['Random_Noise_Scale'] = np.arange(0, 0.25, 0.05)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Random_Noise_Scale'))

# Number of outcomes
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'num_outcomes'
new_df = pd.concat([new_row] * 6, axis=1).transpose()
new_df['Num_Outcomes'] = np.arange(10, 70, 10)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Num_Outcomes'))

# Noise Co-Occurrence Mean Tests
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'co_occur_noise_mean2'
new_df = pd.concat([new_row] * 10, axis=1).transpose()
new_df['Co_Occurrence_Noise_Mean'] = np.arange(0.05, 0.55, 0.05)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Co_Occurrence_Noise_Mean'))

# Number of noise groups
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'num_noise_groups2'
new_df = pd.concat([new_row] * 10, axis=1).transpose()
new_df['Num_Noise_Groups'] = np.arange(10, 110, 10)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Num_Noise_Groups'))

# Random noise scale
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'random_scale2'
new_df = pd.concat([new_row] * 10, axis=1).transpose()
new_df['Random_Noise_Scale'] = np.arange(0.1, 1.1, 0.1)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Random_Noise_Scale'))

# Positive Control
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'positive_control'
new_df = pd.concat([new_row] * 2, axis=1).transpose()
new_df['Num_Noise_Groups'] = 0
new_df['Random_Noise_Scale'] = 0

to_concat.append(pd.concat([new_df] * 10))

# Negative Control
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'negative_control'
new_df = pd.concat([new_row] * 2, axis=1).transpose()

to_concat.append(pd.concat([new_df] * 10))

# Test single outcome
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'single_outcome'
new_df = pd.concat([new_row] * 4, axis=1).transpose()
new_df['Max_Outcome_Prev'] = [0.005, 0.01, 0.05, 0.1]
new_df['Num_Outcomes'] = 1
to_concat.append(pd.concat([new_df] * 10).sort_values(by='Max_Outcome_Prev'))

# Number of Codes
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'num_codes'
new_df = pd.concat([new_row] * 10, axis=1).transpose()
new_df['Total_Codes'] = np.arange(150, 1510, 150)

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Total_Codes'))

# Number of People
new_row = pd.Series(defaults)
new_row.loc['Sim_ID'] = 'population_size'
new_df = pd.concat([new_row] * 7, axis=1).transpose()
new_df['Num_People'] = [500, 1000, 5000, 10000, 50000, 100000, 150000]

to_concat.append(pd.concat([new_df] * 10).sort_values(by='Num_People'))

trials = pd.concat(to_concat, ignore_index=True)

for k, v in fixed_parameters.items():
    trials[k] = v

trials['Seed_Offset'] = trials.index % 10
trials.index.name = 'Row_Number'

trials.loc[trials['Sim_ID'] == 'negative_control', 'Min_Signpost_Codes'] = 0
trials.loc[trials['Sim_ID'] == 'negative_control', 'Max_Signpost_Codes'] = 0

for id, subDF in trials.groupby('Sim_ID'):
    subDF.to_csv('/project/ritchie_scratch/lindsay/Rel_Risk_Pankhuri/Simulations_Scaling/Trial_Tables/' + id + '_trial_info.csv')