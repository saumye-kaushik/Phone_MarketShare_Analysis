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
import matplotlib as plt
import seaborn as sb


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
def mshare(input_df):
    share_score_list = []
    for i in range(1, 6):
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
def rnd_weight(input_df):
    rnd_score_list = []
    for i in range(1, 6):
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
def profitmargin_weight(input_df):
    profitmargin_score_list = []
    for i in range(1, 6):
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


#Function for Revenue weightage
def revenue_weight(input_df):
    revenue_score_list = []
    for i in range(1, 6):
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

# Function for simulating service partnership, if service partnership increases it shall give the company a postive boost, if the service
# partnership remains the same or decreases it will affect the company negatively.
def service_partnership():
    sp = [True]
    i = 1
    weight = 0

    while i < 10:
        seq = bool(random.getrandbits(1))

        i += 1

        if seq and sp[-1] is False:
            weight += 300

        elif seq is False and sp[-1] is False:
            weight += -50

        elif seq is False and sp[-1] is True:
            weight += -150
        sp.append(seq)
    return weight

# Function for simulating new inventions, if new inventions are made it will have a positive impact on the company, if no new invention
#  is made for two continuous year it will affect the company negatively.
def new_invention():
    list_1 = []
    i = 1
    weight = 0

    while i < 10:
        seq = bool(random.getrandbits(1))
        list_1.append(seq)
        i += 1

        if seq:
            list_1.append(False)
            i += 1
            weight += 250

        elif seq is False and list_1[-1] is False:
            weight += -250

        elif seq is False and list[-1] is True:
            weight += 0

    return weight

# Function for simulating number of countries company is active in, if company is active in more than 45 countires it will have a positive
# impact on the company, if active country is less than 45 it will affect the company negatively.
def active_countries():
    weight_ls = []

    for i in range(10):
        act_var = random.randint(30, 60)
        if act_var >= 45:
            weight_ls.append(200)
        else:
            weight_ls.append(50)

        avg_weight = calculate_average(weight_ls)

    return avg_weight


# Function for MC simulation
def calculate_yoy_weight(prev_weight_score: float = 0, ac_score: float = 0, sp_score: float = 0,
                         ni_score: float = 0) -> float:

    company_sim_weightage = prev_weight_score + ac_score + sp_score + ni_score
    return company_sim_weightage


def get_company_scores(company_prev_score: float):

    company_ac_score = active_countries()
    company_sp_score = service_partnership()
    company_ni_score = new_invention()

    company_mc_score = calculate_yoy_weight(company_prev_score,  company_ac_score, company_sp_score,
                                            company_ni_score)
    return company_mc_score


if __name__ == '__main__':
    company_list = ['Samsung', 'Apple', 'LG', 'Huawei']

    marketshare_df = input_file('input/Marketshare.csv')
    profitmargin_df = input_file('input/ProfitMargin.csv')
    revenue_df = input_file('input/Revenue.csv')
    rndexpenditure_df = input_file('input/RnDExpenditure.csv')

    for company in company_list:
        company_ms_score = mshare(marketshare_df[marketshare_df['Company'] == company])
        company_rnd_score = rnd_weight(rndexpenditure_df[rndexpenditure_df['Company'] == company])
        company_profitmargin_score = profitmargin_weight(profitmargin_df[profitmargin_df['Company'] == company])
        company_revenue_score = revenue_weight(revenue_df[revenue_df['Company'] == company])

        previous_score_list = [company_ms_score, company_rnd_score, company_profitmargin_score, company_revenue_score]
        company_previous_score = calculate_previous_data_weight(previous_score_list)

        for i in range(5):
            company_score_list = []
            for j in range(0, 1000):
                company_score_sim = get_company_scores(company_previous_score)
                company_score_list.append(company_score_sim)

            company_score_array = np.array(company_score_list)
            company_score_yearly = np.mean(company_score_array)
            company_previous_score = company_score_yearly
            print(i)
            print(company)
            print(company_score_yearly)
        print('-'*40)
