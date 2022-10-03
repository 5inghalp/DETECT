import numpy as np

configfile: 'config_relrisk_converge.yaml'

wildcard_constraints:
    code_type='(icd|phecode)'

rule all:
    input:
        expand('Trajectories/trajectories_I10_{outcome}_phecode_{dataset}.csv', outcome=[c.replace('*', '') for c in open(config['phecode_list']).read().splitlines()], dataset=[k for k, v in config['phecode_date_matrices'].items()]),
        expand('Trajectories/trajectories_{index}_{outcome}_icd_{dataset}.csv', outcome=[c.replace('*', '') for c in open(config['icd_code_list']).read().splitlines()], dataset=[k for k, v in config['icd_date_matrices'].items()], index=config['index_code_list']),
        expand('Trajectory_Times/{index}_{outcome}_icd_4dig.all_signpost_times.csv.gz', outcome=['I63.', 'I21.', 'N17.9', 'N18.9'])

rule map_icd_code_sets:
    output:
        map='Code_Maps/icd_{dataset}.json'
    input:
        codes=config['icd_code_list'],
        date_mtx=lambda wildcards: config['icd_date_matrices'][wildcards.dataset]
    resources: mem_mb=20000
    run:
        import json
        import pandas as pd

        df_head = pd.read_csv(input.date_mtx, nrows=5, index_col='IID')
        print(df_head)

        codes = open(input.codes).read().splitlines()
        print(codes)

        col_codes = df_head.columns.to_series()
        code_map = {}
        for c in codes:
            code_map[c.replace('*', '')] = list(col_codes[col_codes.str.match(c)].values)
        print(code_map)

        json.dump(code_map, open(output.map, 'w+'))

rule map_phecode_sets:
    output:
        map='Code_Maps/phecode_{dataset}.json'
    input:
        codes=config['phecode_list'],
        date_mtx=lambda wildcards: config['phecode_date_matrices'][wildcards.dataset]
    resources: mem_mb=20000
    run:
        import json
        import pandas as pd

        df_head = pd.read_csv(input.date_mtx, nrows=5, index_col='IID')
        print(df_head)

        codes = open(input.codes).read().splitlines()
        codes = ['Phe' + c for c in codes]
        print(codes)

        col_codes = df_head.columns.to_series()
        code_map = {}
        for c in codes:
            code_map[c.replace('*', '')] = list(col_codes[col_codes.str.match(c)].values)
        print(code_map)

        json.dump(code_map, open(output.map, 'w+'))

rule test_outcome:
    output:
        results='RR_Tests/{index}_{outcome}_{code_type}_{dataset}.csv'
    input:
        date_mtx=lambda wildcards: config[wildcards.code_type + '_date_matrices'][wildcards.dataset],
        map='Code_Maps/{code_type}_{dataset}.json'
    params:
        index_code=lambda wildcards: wildcards.index
    resources: mem_mb=20000
    run:
        import json
        import pandas as pd
        from scipy.stats import norm

        df = pd.read_csv(input.date_mtx, nrows=None, index_col='IID')
        code_map = json.load(open(input.map))

        print(df)
        print(code_map)
        outcome = 'Phe' + wildcards.outcome if wildcards.code_type == 'phecode' else wildcards.outcome
        code_list = code_map[outcome]
        df = df[~pd.isnull(df[params.index_code])]
        outcome_series = df[code_list].min(axis=1)
        df = df.drop(columns=code_list)
        outcome_series.name = outcome
        print(df)

        # Do testing in 2D for efficiency
        # Rows = codes, Columns = people

        # 2D matrix of A
        cast_shape = df.transpose().shape
        print(cast_shape)
        A_times = np.broadcast_to(df[params.index_code], cast_shape)
        non_nan_A = ~np.isnan(A_times)
        print(A_times)
        # 2D matrix of outcome/C
        C_times = np.broadcast_to(outcome_series, cast_shape)
        non_nan_C = ~np.isnan(C_times)
        print(C_times)
        # 2D matrix of intermediate code rows by people
        B_times = df.transpose().values
        non_nan_B = ~np.isnan(B_times)
        print(B_times)

        non_nan_ABC = non_nan_A & non_nan_B & non_nan_C

        # Upper left box
        ABC = (A_times <= B_times) & (B_times <= C_times) & non_nan_ABC
        BAC = (B_times <= A_times) & (A_times <= C_times) & non_nan_ABC
        caseB_caseC = ABC | BAC

        # Upper right box
        ACB = (A_times <= C_times) & (C_times <= B_times) & non_nan_ABC
        AC_not_B = (A_times <= C_times) & ~non_nan_B
        contB_caseC = ACB | AC_not_B

        # Lower left box
        BA_not_C = (B_times <= A_times) & ~non_nan_C
        AB_not_C = (A_times <= B_times) & ~non_nan_C
        BAC = (B_times <= A_times) & (A_times <= C_times) & non_nan_ABC
        caseB_contC = BA_not_C | AB_not_C | BAC

        # Lower right box
        CAB = (C_times <= A_times) & (A_times <= B_times) & non_nan_ABC
        CBA = (C_times <= B_times) & (B_times <= A_times) & non_nan_ABC
        CA_not_B = (C_times <= A_times) & ~non_nan_B
        A_not_B_not_C = ~non_nan_B & ~non_nan_C
        contB_contC = CAB | CBA | CA_not_B | A_not_B_not_C

        results = pd.DataFrame(index=df.columns)
        results['caseB_caseC'] = np.count_nonzero(caseB_caseC, axis=1)
        results['contB_caseC'] = np.count_nonzero(contB_caseC, axis=1)
        results['caseB_contC'] = np.count_nonzero(caseB_contC, axis=1)
        results['contB_contC'] = np.count_nonzero(contB_contC, axis=1)


        results['numerator'] = results['caseB_caseC'] / (results['caseB_caseC'] + results['contB_caseC'])
        results['denominator'] = results['caseB_contC'] / (results['caseB_contC'] + results['contB_contC'])
        results['RR'] = results['numerator'] / results['denominator']

        results['LN_RR'] = np.log(results['RR'])
        four_cols = ['caseB_caseC', 'contB_caseC', 'caseB_contC', 'contB_contC']
        results['SE_LN_RR'] = (results[four_cols].astype(float) ** -1).sum(axis=1) ** 0.5
        results['P'] = 2 * (1 - norm.cdf(abs(results['LN_RR']), loc=0, scale=results['SE_LN_RR']))
        results['sum_check'] = results[four_cols].sum(axis=1)

        results.index.name = 'signpost'

        print(results)
        results.to_csv(output.results)

rule make_filtered_trajectories:
    output:
        trajectories='Trajectories/trajectories_{index}_{outcome}_{code_type}_{dataset}.csv',
        keep_codes='Signpost_Codes/{index}_{outcome}_{code_type}_{dataset}.txt',
        keep_code_results='Signpost_Codes/{index}_{outcome}_{code_type}_{dataset}_results.csv'
    input:
        results = 'RR_Tests/{index}_{outcome}_{code_type}_{dataset}.csv',
        map= 'Code_Maps/{code_type}_{dataset}.json',
        date_mtx = lambda wildcards: config[wildcards.code_type + '_date_matrices'][wildcards.dataset]
    params:
        index_code = lambda wildcards: wildcards.index
    resources: mem_mb=20000
    run:
        import pandas as pd
        import json

        outcome = 'Phe' + wildcards.outcome if wildcards.code_type == 'phecode' else wildcards.outcome

        results = pd.read_csv(input.results, index_col='signpost')
        results = results[(results['RR'] > 1) | np.isinf(results['RR'])]
        results = results[(results['P'] <= 0.05) | np.isinf(results['RR'])]
        print(results)

        df = pd.read_csv(input.date_mtx, index_col='IID', nrows=None)
        code_map = json.load(open(input.map))
        print(code_map)
        code_list = code_map[outcome]
        outcome_series = df[code_list].min(axis=1)
        df[outcome] = outcome_series

        keep_cols = list(results.index)
        keep_cols.append(params.index_code)
        print(keep_cols)
        print(df[keep_cols])
        keep_cols.append(outcome)
        print(df[keep_cols])

        trajectories = df[keep_cols].apply(lambda x: '->'.join(x.dropna().astype(str).sort_values().index), axis=1)
        trajectories.name = 'Filtered_Trajectory'
        print(trajectories)

        outcome_mtx = np.broadcast_to(outcome_series.values, df.transpose().shape).T
        print(outcome_mtx)
        pre_outcome_df = df.mask((df > outcome_mtx) | np.isnan(outcome_mtx))
        print(pre_outcome_df)
        print(pre_outcome_df.count(axis=1))
        trajectories2 = pre_outcome_df[keep_cols].apply(lambda x: '->'.join(x.dropna().astype(str).sort_values().index), axis=1)
        trajectories2.name = 'Filtered_Trajectory_Pre_Outcome'

        full_trajectories = df.apply(lambda x: '->'.join(x.dropna().astype(str).sort_values().index),axis=1)
        full_trajectories.name = 'Full_Trajectory'
        print(full_trajectories)

        all_trajectories = pd.concat([trajectories, trajectories2, full_trajectories], axis=1)
        print(all_trajectories)

        all_trajectories['Filtered_Code_Count'] = df[keep_cols].count(axis=1)
        all_trajectories['Filtered_Pre_Outcome_Code_Count'] = pre_outcome_df[keep_cols].count(axis=1)
        all_trajectories['Full_Code_Count'] = df.count(axis=1)

        print(all_trajectories)

        open(output.keep_codes, 'w+').write('\n'.join(results.index) + '\n')
        results.to_csv(output.keep_code_results)
        all_trajectories.to_csv(output.trajectories)

rule compute_data_span:
    output:
        info = '4digit_icd_data_span.csv',
        info_plot = '4digit_icd_data_span_hist.png'
    input:
        date_mtx = 'Date_Matrices/4digit_thresh15_053122.final.RR.condition_matrix.csv.gz'
    resources: mem_mb=20000
    run:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        dm = pd.read_csv(input.date_mtx,nrows=int(1E4),index_col='IID')

        data_span_info = pd.concat([dm.min(axis=1), dm.max(axis=1)],axis=1)
        print(data_span_info)
        data_span_info.columns = ['Earliest', 'Latest']
        data_span_info['Years_of_Data'] = data_span_info['Latest'] - data_span_info['Earliest']
        print(data_span_info)
        print(data_span_info['Years_of_Data'].describe())

        plt.hist(data_span_info['Years_of_Data'])
        plt.xlabel('Years of EHR Data')
        plt.ylabel('People Count')
        plt.title('Distribution of EHR Data Lengths Per Person')
        plt.gcf().set_size_inches(8,6)
        plt.gcf().set_facecolor('w')
        plt.savefig(output.info_plot,dpi=120,bbox_inches='tight')

        data_span_info.to_csv(output.info)

rule make_lab_dates:
    output:
        minmax='Lab_Processing/min_max_date_values_{lab}.csv',
        long_table='Lab_Processing/date_values_long_{lab}.csv'
    input:
        labs='Lab_Data/{lab}.csv',
        date_mtx='Date_Matrices/4digit_thresh15_053122.final.RR.condition_matrix.csv.gz'
    resources: mem_mb=20000
    run:
        import pandas as pd
        pd.options.mode.chained_assignment = None
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime

        df = pd.read_csv(input.labs, parse_dates=['ORDER_DATE_SHIFT', 'RESULT_DATE_SHIFT'])
        print(df)
        print(df.columns)
        print(df['ABNORMAL'])
        print(df['ABNORMAL'].value_counts())
        print(df['ORDER_DATE_SHIFT'])
        print(df['RESULT_DATE_SHIFT'])
        print(df['RESULT_DATE_SHIFT'].count())
        df['RESULT_DATE_SHIFT'] = pd.to_datetime(df['RESULT_DATE_SHIFT'], errors='coerce')
        print(df)

        print(df['person_id'].value_counts())

        abnormal_df = df.dropna(subset='ABNORMAL')
        abnormal_df['DATE_VAL'] = (abnormal_df['RESULT_DATE_SHIFT'] - datetime(1970, 1, 1)).apply(lambda x: x.days / 365.25)
        print(abnormal_df)

        dm = pd.read_csv(input.date_mtx, nrows=None, index_col='IID')

        data_span_info = pd.concat([dm.min(axis=1), dm.max(axis=1)], axis=1)
        print(data_span_info)
        data_span_info.columns = ['Earliest', 'Latest']
        data_span_info['Years_of_Data'] = data_span_info['Latest'] - data_span_info['Earliest']
        print(data_span_info)
        print(data_span_info['Years_of_Data'].describe())

        lab_people = dm[dm.index.isin(abnormal_df['person_id'])]
        abnormal_df = abnormal_df[abnormal_df['person_id'].isin(lab_people.index)]

        abnormal_df['After_First_ICD'] = abnormal_df['DATE_VAL'] >= (data_span_info.loc[abnormal_df['person_id'], 'Earliest'].values - 0.5)
        abnormal_df['Before_Last_ICD'] = abnormal_df['DATE_VAL'] <= (data_span_info.loc[abnormal_df['person_id'], 'Latest'].values + 0.5)
        abnormal_df['Within_ALL_ICD'] = abnormal_df['After_First_ICD'] & abnormal_df['Before_Last_ICD']
        abnormal_df = abnormal_df.sort_values(by='DATE_VAL')
        print(abnormal_df)

        new_lab_tables = []

        for person, row in lab_people.iterrows():
            row = row.dropna()
            person_labs = abnormal_df[abnormal_df['person_id'] == person].copy()
            person_min, person_max = data_span_info.loc[person, 'Earliest'], data_span_info.loc[person, 'Latest']

            mtx_dim = (len(person_labs), len(row))

            codes_as_cols = np.broadcast_to(row.values,mtx_dim)

            labs_as_rows = np.broadcast_to(person_labs['DATE_VAL'].values, reversed(mtx_dim)).T

            codes_as_cols_minus6 = codes_as_cols - 0.5
            codes_as_cols_plus6 = codes_as_cols + 0.5

            after_start_window = labs_as_rows >= codes_as_cols_minus6
            before_end_window = labs_as_rows <= codes_as_cols_plus6
            within_window = before_end_window & after_start_window

            person_labs['Close_to_any_ICD'] = np.any(within_window, axis=1)
            person_labs['Close_to_other_abnormal'] = False
            first_to_penultimate = person_labs['DATE_VAL'].iloc[:-1].values
            second_to_end = person_labs['DATE_VAL'].iloc[1:].values
            second_to_penultimate = person_labs['DATE_VAL'].iloc[1:-1].values
            third_to_end = person_labs['DATE_VAL'].iloc[2:].values
            person_labs['Close_to_other_abnormal'].iloc[:-1] |= (second_to_end - first_to_penultimate) <= 1
            person_labs['Close_to_other_abnormal'].iloc[1:] |= (second_to_end - first_to_penultimate) <= 1

            new_lab_tables.append(person_labs)

        new_labs = pd.concat(new_lab_tables)
        print(new_labs)

        new_labs = new_labs[new_labs['Close_to_any_ICD']]
        new_labs = new_labs[new_labs['Close_to_other_abnormal']]

        print(new_labs)

        person_dates = pd.concat([new_labs.drop_duplicates(subset=['person_id'], keep='first').set_index('person_id')['DATE_VAL'], new_labs.drop_duplicates(subset=['person_id'], keep='last').set_index('person_id')['DATE_VAL']], axis=1)
        person_dates.columns = [wildcards.lab + '_FIRST', wildcards.lab + '_LAST']
        person_dates.index.name = 'IID'
        print(person_dates)

        person_dates.to_csv(output.minmax)

        new_labs.to_csv(output.long_table, index=False)

rule get_trajectory_times:
    output:
        times='Trajectory_Times/{index}_{outcome}_{code_type}_{dataset}.all_signpost_times.csv.gz',
        indiv_times='Trajectory_Times/{index}_{outcome}_{code_type}_{dataset}.individual_signpost_times.csv.gz',
        signpost_freq_times='Trajectory_Times/{index}_{outcome}_{code_type}_{dataset}_sumstats_by_freq.csv',
        signpost_RR_times='Trajectory_Times/{index}_{outcome}_{code_type}_{dataset}_sumstats_by_RR.csv',
        pie_charts=expand('Pie_Charts/{{index}}_{{outcome}}_{{code_type}}_{{dataset}}_{rank}_{i}.png', i=np.arange(1, 21), rank=['Freq', 'RR']),
        time_scatter_freq='Plots/{index}_{code_type}_{dataset}_signpost_to_{outcome}_times_by_race_Freq.png',
        time_scatter_rr='Plots/{index}_{code_type}_{dataset}_signpost_to_{outcome}_times_by_race_RR.png'
    input:
        trajectories='Trajectories/trajectories_{index}_{outcome}_{code_type}_{dataset}.csv',
        keep_code_results='Signpost_Codes/{index}_{outcome}_{code_type}_{dataset}_results.csv',
        map= 'Code_Maps/{code_type}_{dataset}.json',
        date_mtx=lambda wildcards: config[wildcards.code_type + '_date_matrices'][wildcards.dataset],
        demo=config['demographics_file']
    params:
        index_code=lambda wildcards: wildcards.index
    resources: mem_mb = 15000
    run:
        import pandas as pd
        import numpy as np
        import json
        import sys
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from copy import copy

        results = pd.read_csv(input.keep_code_results)
        results = results[results['signpost'] != 'IMO00']
        date_mtx = pd.read_csv(input.date_mtx, index_col='IID', nrows=None)

        print(results)
        print(date_mtx)
        outcome = 'Phe' + wildcards.outcome if wildcards.code_type == 'phecode' else wildcards.outcome

        trajectories = pd.read_csv(input.trajectories, index_col='IID')

        code_map = json.load(open(input.map))
        code_list = code_map[outcome]
        outcome_series = date_mtx[code_list].min(axis=1)
        date_mtx[outcome] = outcome_series

        trajectories = trajectories.dropna(subset=['Filtered_Trajectory_Pre_Outcome'])
        print('People with outcome:', len(trajectories))
        trajectories = trajectories[trajectories['Filtered_Pre_Outcome_Code_Count'] > 1]
        print('People with outcome and one or more preceding codes:', len(trajectories))
        trajectories = trajectories[trajectories['Filtered_Trajectory_Pre_Outcome'].str.contains(params.index_code)]
        print('People with ' + params.index_code + ' preceding outcome:', len(trajectories))
        date_mtx = date_mtx.loc[trajectories.index]
        time_index_to_outcome = date_mtx[outcome] - date_mtx[params.index_code]
        quadrant1_ppl = trajectories.index[trajectories['Filtered_Pre_Outcome_Code_Count'] > 2]
        quadrant2_ppl = trajectories.index[trajectories['Filtered_Pre_Outcome_Code_Count'] == 2]
        print('Quadrant 1 people:', len(quadrant1_ppl))
        print('Quadrant 2 people:', len(quadrant2_ppl))
        sumstats = pd.concat([time_index_to_outcome.describe(), time_index_to_outcome.loc[quadrant1_ppl].describe(), time_index_to_outcome.loc[quadrant2_ppl].describe()], axis=1)
        sumstats.columns = ['Top Row (All A->C)', 'Quadrant 1 ([A&B]->C)', 'Quadrant 2 (A->C no B)']
        print('Summary Statistics for Time from ' + params.index_code + ' to Outcome')
        print(sumstats)
        print(trajectories['Filtered_Trajectory_Pre_Outcome'])
        print(trajectories['Filtered_Pre_Outcome_Code_Count'])
        date_mtx = date_mtx.loc[quadrant1_ppl]
        trajectories = trajectories.loc[quadrant1_ppl]

        trajectory_nodes = trajectories['Filtered_Trajectory_Pre_Outcome'].str.split('->', expand=True)
        node_cols = trajectory_nodes.columns
        trajectory_nodes['IID'] = trajectory_nodes.index
        trajectory_nodes['Mask'] = False
        print(trajectory_nodes)

        signpost_codes = results['signpost']
        date_mtx = date_mtx[signpost_codes]
        outcome_series = outcome_series.loc[quadrant1_ppl]
        print(outcome_series)

        # Mask times after outcome
        outcome_mtx = np.broadcast_to(outcome_series, date_mtx.transpose().shape).T
        date_mtx = date_mtx.mask(date_mtx > outcome_mtx)

        signpost_to_outcome_times = outcome_mtx - date_mtx
        print(signpost_to_outcome_times)

        demo = pd.read_csv(input.demo, index_col='person_id')
        demo = demo.loc[quadrant1_ppl]
        demo['race_source_value'] = demo['race_source_value'].mask(~demo['race_source_value'].isin(['White', 'Black', 'Asian'])).fillna('Other')
        print(demo['race_source_value'].value_counts())

        signpost_to_outcome_sumstats = signpost_to_outcome_times.describe().transpose()
        signpost_to_outcome_sumstats['RR'] = results['RR'].values
        signpost_to_outcome_sumstats.index.name = 'signpost'
        print(signpost_to_outcome_sumstats)

        white_sumstats = signpost_to_outcome_times.loc[demo[demo['race_source_value'] == 'White'].index].describe().transpose()
        black_sumstats = signpost_to_outcome_times.loc[demo[demo['race_source_value'] == 'Black'].index].describe().transpose()
        asian_sumstats = signpost_to_outcome_times.loc[demo[demo['race_source_value'] == 'Asian'].index].describe().transpose()
        other_sumstats = signpost_to_outcome_times.loc[demo[demo['race_source_value'] == 'Other'].index].describe().transpose()

        signpost_to_outcome_sumstats[['W-Mean', 'W-Median', 'W-Count']] = white_sumstats[['mean', '50%', 'count']].values
        signpost_to_outcome_sumstats[['B-Mean', 'B-Median', 'B-Count']] = black_sumstats[['mean', '50%', 'count']].values
        signpost_to_outcome_sumstats[['A-Mean', 'A-Median', 'A-Count']] = asian_sumstats[['mean', '50%', 'count']].values
        signpost_to_outcome_sumstats[['O-Mean', 'O-Median', 'O-Count']] = other_sumstats[['mean', '50%', 'count']].values

        col_order = ['RR']
        col_order.append('count')
        col_order.extend([x + '-Count' for x in ['W', 'B', 'A', 'O']])
        col_order.append('mean')
        col_order.extend([x + '-Mean' for x in ['W', 'B', 'A', 'O']])
        col_order.append('50%')
        col_order.extend([x + '-Median' for x in ['W', 'B', 'A', 'O']])
        signpost_to_outcome_sumstats = signpost_to_outcome_sumstats[col_order]
        print(signpost_to_outcome_sumstats)

        signpost_to_outcome_sumstats = signpost_to_outcome_sumstats.sort_values(by='count', ascending=False)

        freq_codes = []
        test_index = 0
        while len(freq_codes) < 20:
            test_code = signpost_to_outcome_sumstats.index[test_index]
            skip = False
            for c in freq_codes:
                if test_code[:3] == c[:3]:
                    print(test_code, c, 'Skip')
                    skip = True
            if not skip:
                freq_codes.append(test_code)
            test_index += 1
        print(signpost_to_outcome_sumstats.index[:20])
        print(freq_codes)


        freq_sumstats = signpost_to_outcome_sumstats.loc[freq_codes]
        print(freq_sumstats)
        freq_sumstats.to_csv(output.signpost_freq_times)

        signpost_to_outcome_sumstats = signpost_to_outcome_sumstats.sort_values(by='RR', ascending=False)

        rr_codes = []
        test_index = 0
        while len(rr_codes) < 20:
            test_code = signpost_to_outcome_sumstats.index[test_index]
            skip = False
            if test_code == 'I69.3':
                skip = True
            if signpost_to_outcome_sumstats.loc[test_code, 'count'] < 50:
                skip = True
            for c in rr_codes:
                if test_code[:3] == c[:3]:
                    print(test_code, c, 'Skip')
                    skip = True
            if not skip:
                rr_codes.append(test_code)
            test_index += 1

        print(signpost_to_outcome_sumstats.index[:20])
        print(rr_codes)

        rr_sumstats = signpost_to_outcome_sumstats.loc[rr_codes]
        print(rr_sumstats)
        rr_sumstats.to_csv(output.signpost_RR_times)

        signpost_to_outcome_times.to_csv(output.indiv_times)
        signpost_to_outcome_sumstats.to_csv(output.times)

        cmap = plt.cm.get_cmap('Blues', 6)
        color_dict = dict(zip(['White', 'Black', 'Other', 'Asian'], [cmap(i + 2) for i in range(4)]))

        # Make pie charts
        for i in range(20):
            for rank in ['Freq', 'RR']:
                output_file = '_'.join(['Pie_Charts/' + outcome, wildcards.code_type, wildcards.dataset, rank,  str(i+1) + '.png'])
                print(output_file)
                use_df = freq_sumstats if rank == 'Freq' else rr_sumstats
                code = use_df.index[i]
                code_demo = demo.loc[date_mtx[code].dropna().index]
                race_counts = code_demo['race_source_value'].value_counts()
                plt.pie(x=race_counts.values, labels=race_counts.index, colors=[color_dict[r] for r in race_counts.index], startangle=90)
                plt.gcf().set_facecolor('w')
                plt.gcf().set_size_inches(6,6)
                plt.title(code)
                plt.savefig(output_file, dpi=150)
                plt.clf()

        code_remap = {'E11.9': 'T2D',
                      'N52.9': 'ED',
                      'M10.9': 'Gout',
                      'E78.5': 'Hyperlipidemia',
                      'J44.9': 'COPD',
                      'H26.9': 'Cataract',
                      'E87.5': 'Hyperkalemia',
                      'I20.9': 'Angina Pectoris',
                      'I48.9': 'A-Fib',
                      'I65.2': 'Carotid Occ\'n',
                      'I73.9': 'Vascular Disease',
                      'I25.1': 'Atherosclerosis',
                      'G62.9': 'Polyneuropathy',
                      'N18.3': 'CKD, Stage 3',
                      'I50.9': 'Heart Failure',
                      'R06.0': 'Dyspnea',
                      'R07.9': 'Chest Pain',
                      'R29.8': 'Other Physical Symptoms',
                      'I63.9': 'Stroke',
                      'N17.9': 'Acute Renal Failure',
                      'K21.9': 'Acid Reflux',
                      'N39.0': 'UTI',
                      'R60.9': 'Edema',
                      'D64.9': 'Anemia',
                      'M54.9': 'Dorsalgia',
                      'E87.6': 'Hypokalemia',
                      'H40.0': 'Glaucoma',
                      'H90.3': 'Hearing Loss',
                      'R42': 'Dizziness',
                      'K59.0': 'Constipation',
                      'R63.4': 'Weight Loss',
                      'H25.1': 'Nuclear Cataract',
                      'R55': 'Syncope and Collapse',
                      'G45.9': 'TIA',
                      'A08.2': 'Intestinal Infection',
                      'N25.9': 'Renal Tube Dysfunction',
                      'E16.8': 'Pancreatic Disorder',
                      'G70.0': 'Myasthenia Gravis',
                      'E11.3': 'T2D',
                      'N18.5': 'CKD, Stage 5',
                      'I35.9': 'Aortic Valve Disorder',
                      'I25.2': 'Myocardial Infarction',
                      'J44.1': 'COPD',
                      'L08.9': 'Subcutaneous Infection',
                      'I42.9': 'Cardiomyopathy',
                      'I20.0': 'Angina',
                      'M15.9': 'Polyosteoarthritis',
                      'E10.9': 'T1D',
                      'I67.8': 'Cerebrovascular Disease',
                      'H40.1': 'Open-Angle Glaucoma',
                      'R32': 'Incontinence',
                      'I70.2': 'Atherosclerosis',
                      'E16.2': 'Hypoglycemia',
                      'G40.9': 'Epilepsy',
                      'R56.9': 'Convulsions',
                      'R26.9': 'Abnormalities of Gait',
                      'F03.9': 'Dementia',
                      'H53.4': 'Visual Field Defects',
                      'R41.8': 'Cognitive Abnormality',
                      'R53.1': 'Abnormal Lab Findings',
                      'I69.3': 'Sequelae of Cerebral Infarction'}

        # Make time scatter plots
        for rank in ['Freq', 'RR']:
            test = freq_sumstats if rank == 'Freq' else rr_sumstats
            test = test.sort_values(by='mean', ascending=False).copy()
            test = test.rename(index=code_remap)

            plt.gcf().set_size_inches(10.2, 5)

            cmap = plt.cm.get_cmap('viridis',5)

            plt.scatter(np.arange(20),test['mean'],s=test['count'],label='All', color=cmap(4))
            plt.scatter(np.arange(20),test['W-Mean'],s=test['W-Count'],label='White', color=cmap(3))
            plt.scatter(np.arange(20),test['B-Mean'],s=test['B-Count'],label='Black', color=cmap(2))
            plt.scatter(np.arange(20),test['A-Mean'],s=test['A-Count'],label='Asian', color=cmap(1))
            plt.scatter(np.arange(20),test['O-Mean'],s=test['O-Count'],label='Other', c='mediumpurple')
            plt.gca().set_facecolor('silver')
            plt.xticks(np.arange(20), labels=test.index, rotation=30, ha='right')
            plt.ylim(0, 9)

            for i in range(20):
                plt.axvline(x=i,c='w',zorder=-10)

            plt.legend()

            handles = plt.gca().get_legend().legendHandles
            for h in handles:
                h.set_sizes([20])

            handles.append(copy(handles[0]))
            handles.append(copy(handles[0]))

            print(handles)

            for h in handles[-2:]:
                h.set_color('k')

            handles[-2].set_label('N = ' + str(test['count'].min() if test['count'].min() > 0 else 1))
            handles[-2].set_sizes([test['count'].min()])
            handles[-1].set_label('N = ' + str(test['count'].max()))
            handles[-1].set_sizes([test['count'].max()])
            plt.legend(handles=handles, borderpad=1.5, ncol=5)

            plt.gcf().set_facecolor('w')
            plt.ylabel('Years from Code to ' + outcome)
            plt.xlabel('Signpost Code')
            plt.title(None)

            plt.tight_layout()
            plt.savefig(output.time_scatter_freq if rank == 'Freq' else output.time_scatter_rr, dpi=150)
            plt.clf()
