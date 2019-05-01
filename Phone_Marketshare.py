"""
590PR: Monte Carlo Simulation of smartphone marketshare
Team Members: Saumye Kaushik, Shray Mishra

random variables: new_inventions, service_partnership, active_countries

Hypothesis:
1. Change in Market Share of a company should be more with relatively higher number of new inventions in a given timeframe.
2. An increase in service partnerships should have a positive impact on company’s market share.
3. If a company sees a rise in number of countries it is active in, it should affect the market share positively.



Output Simulation:
1. Market Share
2. All input predictions: R&D, Revenue, Profit Margin
3. Random variable changes yoy: New_inventions, Service_partners, active_countries
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

desired_width = 700
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)


def mod_pert_random(low, likely, high, confidence=4, samples=1):
    """Produce random numbers according to the 'Modified PERT'
    distribution.
    Picked up from the class 12 lecture notes.

    :param low: The lowest value expected as possible.
    :param likely: The 'most likely' value, statistically, the mode.
    :param high: The highest value expected as possible.
    :param confidence: This is typically called 'lambda' in literature
                        about the Modified PERT distribution. The value
                        4 here matches the standard PERT curve. Higher
                        values indicate higher confidence in the mode.
                        Currently allows values 1-18
    :param samples: The highest value expected as possible.
    Formulas from "Modified Pert Simulation" by Paulo Buchsbaum.
    """
    # Check minimum & maximum confidence levels to allow:
    if confidence < 1 or confidence > 18:
        raise ValueError('confidence value must be in range 1-18.')

    mean = (low + confidence * likely + high) / (confidence + 2)

    a = (mean - low) / (high - low) * (confidence + 2)
    b = ((confidence + 1) * high - low - confidence * likely) / (high - low)

    beta = np.random.beta(a, b, samples)
    beta = beta * (high - low) + low
    return beta


# Function for reading input
def input_file(file) -> pd.DataFrame:
    '''

    :param file:
    :return:
    '''
    input_df = pd.read_csv(file)
    return input_df


def calculate_average(score_list: list) -> float:
    return sum(score_list)/len(score_list)


# Function for Marketshare weightage
def mshare(input_df: pd.DataFrame, num_years: int = 5):
    share_score_list = []
    for i in range(1, num_years+1):
        share_value = input_df.iloc[:, i].item()
        if share_value >= 20:
            mshare_score = 20
        elif 10 <= share_value < 20:
            mshare_score = 10
        elif share_value < 10:
            mshare_score = 5
        share_score_list.append(mshare_score)

    # print(share_score_list)
    mshare_score_avg = calculate_average(share_score_list)
    return mshare_score_avg


# Function for R&D weightage
def rnd_weight(input_df: pd.DataFrame, num_years: int = 5):
    rnd_score_list = []
    for i in range(1, num_years+1):
        rnd_value = input_df.iloc[:, i].item()
        if rnd_value >= 10:
            rnd_score = 20
        elif 5 <= rnd_value < 10:
            rnd_score = 10
        elif rnd_value < 5:
            rnd_score = 5
        rnd_score_list.append(rnd_score)

    # print(share_score_list)
    rnd_score_avg = calculate_average(rnd_score_list)
    return rnd_score_avg


# Function for profit margin weightage
def profitmargin_weight(input_df: pd.DataFrame, num_years: int = 5):
    profitmargin_score_list = []
    for i in range(1, num_years+1):
        profitmargin_value = input_df.iloc[:, i].item()
        if profitmargin_value >= 40:
            profitmargin_score = 20
        elif 15 <= profitmargin_value < 40:
            profitmargin_score = 10
        elif profitmargin_value < 15:
            profitmargin_score = 5
        profitmargin_score_list.append(profitmargin_score)

    # print(share_score_list)
    profitmargin_score_avg = calculate_average(profitmargin_score_list)
    return profitmargin_score_avg


# Function for Revenue weightage
def revenue_weight(input_df: pd.DataFrame, num_years: int = 5):
    revenue_score_list = []
    for i in range(1, num_years+1):
        revenue_value = input_df.iloc[:, i].item()
        if revenue_value >= 200:
            revenue_score = 20
        elif 50 <= revenue_value < 200:
            revenue_score = 10
        elif revenue_value < 50:
            revenue_score = 5
        revenue_score_list.append(revenue_score)

    # print(share_score_list)
    revenue_score_avg = calculate_average(revenue_score_list)
    return revenue_score_avg


# Function for calculating historical weightage
def calculate_previous_data_weight(score_list: list) -> int:
    weighted_score = score_list[0]*40 + score_list[1]*15 + score_list[2]*15 + score_list[3]*30
    return weighted_score


# Function for simulating service partnership, if service partnership increases it shall give the company a postive
# boost, if the service partnership remains the same or decreases it will affect the company negatively.
def service_partnership(company_name: str, choice: int):
    sp = [True]
    i = 1
    weight = 0
    if choice == 3 and company_name.lower() == 'huawei':
        return 135
    while i < 10:
        seq = bool(random.getrandbits(1))

        i += 1

        if seq and sp[-1] is False:
            weight += 135

        elif seq is False and sp[-1] is False:
            weight += -22

        elif seq is False and sp[-1] is True:
            weight += -68
        sp.append(seq)
    return weight


# Function for simulating new inventions, if new inventions are made it will have a positive impact on the company, if
# no new invention is made for two continuous year it will affect the company negatively.
def new_invention(company_name: str, choice: int):
    list_1 = []
    i = 1
    weight = 0
    if choice == 2 and company_name.lower() == 'huawei':
        return 90
    else:
        prob_success = mod_pert_random(0, 75, 100).item()
        while i < 10:
            seq = bool(random.getrandbits(1))
            list_1.append(seq)
            i += 1

            if seq:
                list_1.append(False)
                i += 1
                weight += 0.9*prob_success

            elif seq is False and list_1[-1] is False:
                weight += -90

            elif seq is False and list[-1] is True:
                weight += 0

        return weight


# Function for simulating number of countries company is active in, if company is active in more than 45 countries it
# will have a positive impact on the company, if active country is less than 45 it will affect the company negatively.
def active_countries(company_name: str, choice: int):
    weight_ls = []
    selected_company = ''

    if choice == 4:
        selected_company = 'huawei'
    for i in range(10):
        if company_name.lower() == selected_company:
            act_var = random.randint(50, 60)
        else:
            act_var = random.randint(30, 60)
        if act_var >= 45:
            weight_ls.append(90)
        else:
            weight_ls.append(22)

        avg_weight = calculate_average(weight_ls)

    return avg_weight, act_var


# Function for MC simulation
def calculate_yoy_weight(prev_weight_score: float = 0, ac_score: float = 0, sp_score: float = 0,
                         ni_score: float = 0) -> float:

    company_sim_weightage = prev_weight_score + ac_score + sp_score + ni_score
    return company_sim_weightage


def get_company_scores(company_prev_score: float, company_name: str, choice: int):

    company_ac_score, country_count = active_countries(company_name, choice)
    company_sp_score = service_partnership(company_name, choice)
    company_ni_score = new_invention(company_name, choice)

    company_mc_score = calculate_yoy_weight(company_prev_score,  company_ac_score, company_sp_score,
                                            company_ni_score)
    return company_mc_score, country_count


def yearly_marketshare(score_df: pd.DataFrame):
    '''

    :param score_df:
    :return:

    '''
    marketshare_pc_df = pd.DataFrame()

    for i in range(2018, 2024):
        samsung_score = score_df.loc['Samsung', str(i)]
        apple_score = score_df.loc['Apple', str(i)]
        lg_score = score_df.loc['LG', str(i)]
        huawei_score = score_df.loc['Huawei', str(i)]

        total_score = sum(list(score_df[str(i)]))

        marketshare_pc_df.loc['Samsung', i] = round(samsung_score/total_score*100, 2)
        marketshare_pc_df.loc['Apple', i] = round(apple_score / total_score*100, 2)
        marketshare_pc_df.loc['LG', i] = round(lg_score / total_score*100, 2)
        marketshare_pc_df.loc['Huawei', i] = round(huawei_score/total_score*100, 2)

    return marketshare_pc_df


def vis(dataframe):
    '''

    :param dataframe:
    :return:
    '''

    sns.set()
    dataframe.T.plot(kind='bar', stacked=True)
    plt.show()
    transposed_df = dataframe.transpose()
    transposed_df.plot.line()
    plt.show()


def test_weights():
    company_list = ['Samsung', 'Apple', 'LG', 'Huawei']

    marketshare_df = input_file('input/Marketshare.csv')
    profitmargin_df = input_file('input/ProfitMargin.csv')
    revenue_df = input_file('input/Revenue.csv')
    rndexpenditure_df = input_file('input/RnDExpenditure.csv')

    for company in company_list:
        print(company)
        company_ms_score = mshare(marketshare_df[marketshare_df['Company'] == company], 3)
        company_rnd_score = rnd_weight(rndexpenditure_df[rndexpenditure_df['Company'] == company], 3)
        company_profitmargin_score = profitmargin_weight(profitmargin_df[profitmargin_df['Company'] == company], 3)
        company_revenue_score = revenue_weight(revenue_df[revenue_df['Company'] == company], 3)

        previous_score_list = [company_ms_score, company_rnd_score, company_profitmargin_score, company_revenue_score]
        company_previous_score = calculate_previous_data_weight(previous_score_list)
        company_score_list = []
        company_country_list = []
        for j in range(0, 1000):
            company_score_sim, company_country_count = get_company_scores(company_previous_score, company)
            company_score_list.append(company_score_sim)
            company_country_list.append(company_country_count)

        company_score_array = np.array(company_score_list)
        company_score_yearly = np.mean(company_score_array)

        company_country_array = np.array(company_country_list)
        company_country_yearly = np.mean(company_country_array)
        print("Previous Year Score: " + str(company_previous_score))
        print("Simulated Score: " + str(company_score_yearly))

        company_ms_score = mshare(marketshare_df[marketshare_df['Company'] == company], 4)
        company_rnd_score = rnd_weight(rndexpenditure_df[rndexpenditure_df['Company'] == company], 4)
        company_profitmargin_score = profitmargin_weight(profitmargin_df[profitmargin_df['Company'] == company], 4)
        company_revenue_score = revenue_weight(revenue_df[revenue_df['Company'] == company], 4)

        previous_score_list = [company_ms_score, company_rnd_score, company_profitmargin_score, company_revenue_score]
        company_next_score = calculate_previous_data_weight(previous_score_list)
        print("Next Year Score: " + str(company_next_score))


def marketshare_sim(choice : int):
    company_list = ['Samsung', 'Apple', 'LG', 'Huawei']

    marketshare_df = input_file('input/Marketshare.csv')
    profitmargin_df = input_file('input/ProfitMargin.csv')
    revenue_df = input_file('input/Revenue.csv')
    rndexpenditure_df = input_file('input/RnDExpenditure.csv')

    df_score_columns = ['2018']
    df_score_yearly = pd.DataFrame(columns=df_score_columns)

    for company in company_list:

        company_ms_score = mshare(marketshare_df[marketshare_df['Company'] == company], 5)
        company_rnd_score = rnd_weight(rndexpenditure_df[rndexpenditure_df['Company'] == company], 5)
        company_profitmargin_score = profitmargin_weight(profitmargin_df[profitmargin_df['Company'] == company], 5)
        company_revenue_score = revenue_weight(revenue_df[revenue_df['Company'] == company], 5)

        previous_score_list = [company_ms_score, company_rnd_score, company_profitmargin_score, company_revenue_score]
        company_previous_score = calculate_previous_data_weight(previous_score_list)

        df_score_yearly.loc[company, '2018'] = company_previous_score

        for i in range(2019, 2024):
            company_score_list = []
            company_country_list = []
            for j in range(0, 1000):
                company_score_sim, company_country_count = get_company_scores(company_previous_score, company, choice)
                company_score_list.append(company_score_sim)
                company_country_list.append(company_country_count)

            company_score_array = np.array(company_score_list)
            company_score_yearly = np.mean(company_score_array)

            df_score_yearly.loc[company, str(i)] = round(company_score_yearly)

            company_previous_score = company_score_yearly
    yearly_split_df = yearly_marketshare(df_score_yearly)
    print('The Market Share simulation for the 4 companies over the next 5 years is -')
    print(yearly_split_df)
    vis(yearly_split_df)

    if choice == 4 and company == 'Huawei':
        company_country_array = np.array(company_country_list)
        company_country_yearly = np.mean(company_country_array)
        print('For year ' + str(i) + ' Number of countries for Huawei is ' + str(round(company_country_yearly)))


if __name__ == '__main__':
    choice = input('Enter 1 for Monte Carlo Simulation, 2 for impact of new inventions in market share, 3 for change in'
                   ' market share for service partner fluctuation, 4 for changes in market share with active countries:'
                   ' ')
    choice = int(choice)
    if choice == 1:
        marketshare_sim(choice)
    elif choice == 2:
        print('Checking for Huawei with more inventions')
        marketshare_sim(choice)
    elif choice == 3:
        print('Checking for Huawei with higher service partnerships')
        marketshare_sim(choice)
    elif choice == 4:
        print('Checking for Huawei with higher country count')
        marketshare_sim(choice)
    else:
        print('Invalid Input, please try again')
