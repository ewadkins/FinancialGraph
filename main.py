import csv
import datetime
from dateutil.parser import parse
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import math
from display_utils import DynamicConsoleTable


###########################################################
#####                  Configuration                  #####
###########################################################

# NOTE: CSV files should include columns of the format:
# date, transaction description, change in balance, balance
#
# The 1st and 4th columns, which are the date and balance
# after the transaction, respecitively, are required.

table_collapse_days = False
plot_collapse_days = True
include_credit = True
output_table_dividers = True

# Each file included below should contain all data for one account.
# So, account transactions shouldn't be split between multiple files,
# and transactions of multiple accounts shouldn't be in the same file.
checking_transactions_filepaths = ['checking.csv']
savings_transactions_filepaths = ['savings.csv']
credit_transactions_filepaths = ['credit1234.csv', 'credit9876.csv']

###########################################################


##### Helper functions

def load_transactions(file_path):
    with open(file_path, 'rU') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        lines = [row for row in reader]
        return map(lambda t: [parse(t[0]), t[1],
                              float(t[2].replace(',', '') if t[2] else 0),
                              float(t[3].replace(',', ''))], lines)

def format_balance(balance):
    return '{0:.2f}'.format(balance)

def format_percent(percent):
    return '{0:.2f}'.format(percent)

def year_fraction(date):
    def since_epoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = since_epoch
    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)
    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration
    return date.year + fraction

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months_full = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
def transform_ticks(ticks):
    month_indices = map(lambda t: int(round(round(t / (1./12), 1) / 12 % 1 * 12, 1)), ticks)
    return [months[month_indices[i]] + \
            ('  ' + str(int(round(ticks[i], 1))) if month_indices[i] == 0 else '')
            for i in range(len(ticks))]

def append_type(type):
    return lambda t: t + [type]


##### Data loading

checking_accounts = map(load_transactions, checking_transactions_filepaths)
savings_accounts = map(load_transactions, savings_transactions_filepaths)
credit_accounts = map(load_transactions, credit_transactions_filepaths)


##### Data processing

if not include_credit:
    credit_accounts = []
    
checking_accounts = [map(append_type(('checking', i)), checking_accounts[i])
                     for i in range(len(checking_accounts))]
savings_accounts = [map(append_type(('savings', i)), savings_accounts[i])
                    for i in range(len(savings_accounts))]
credit_accounts = [map(append_type(('credit', i)), credit_accounts[i])
                   for i in range(len(credit_accounts))]

transactions = sum(checking_accounts + savings_accounts + credit_accounts, [])
transaction_indices = list(range(len(transactions)))
transaction_indices.sort(key=lambda i: transactions[i][0])
transactions = np.array(transactions)[transaction_indices]

total_balances = []
checking_balances = []
savings_balances = []
credit_balances = []
meta = []
checking_balance_array = [0.0] * len(checking_accounts)
savings_balance_array = [0.0] * len(savings_accounts)
credit_balance_array = [0.0] * len(credit_accounts)
last_date = None
for i in range(len(transactions)):
    (date, desc, amount, balance, id) = transactions[i]
    type, account_index = id
    if amount == 0:
        continue
    if last_date:
        combined_checking_balance = sum(checking_balance_array)
        combined_savings_balance = sum(savings_balance_array)
        combined_credit_balance = sum(credit_balance_array)
        total_balances[-1].append((last_date, combined_checking_balance + combined_savings_balance + combined_credit_balance))
        checking_balances[-1].append((last_date, combined_checking_balance))
        savings_balances[-1].append((last_date, combined_savings_balance))
        credit_balances[-1].append((last_date, combined_credit_balance))
        meta[-1].append((desc, id))
    if last_date != date:
        total_balances.append([])
        checking_balances.append([])
        savings_balances.append([])
        credit_balances.append([])
        meta.append([])
    last_date = date
    if type == 'checking':
        checking_balance_array[account_index] = balance
    if type == 'savings':
        savings_balance_array[account_index] = balance
    if type == 'credit':
        credit_balance_array[account_index] = balance
combined_checking_balance = sum(checking_balance_array)
combined_savings_balance = sum(savings_balance_array)
combined_credit_balance = sum(credit_balance_array)
total_balances[-1].append((last_date, combined_checking_balance + combined_savings_balance + combined_credit_balance))
checking_balances[-1].append((last_date, combined_checking_balance))
savings_balances[-1].append((last_date, combined_savings_balance))
credit_balances[-1].append((last_date, combined_credit_balance))
meta[-1].append((desc, id))

collapsed_total_balances = np.array(map(lambda x: x[-1], total_balances))
collapsed_checking_balances = np.array(map(lambda x: x[-1], checking_balances))
collapsed_savings_balances = np.array(map(lambda x: x[-1], savings_balances))
collapsed_credit_balances = np.array(map(lambda x: x[-1], credit_balances))
collapsed_meta = map(lambda x: x[-1], meta)
total_balances = np.concatenate(total_balances)
checking_balances = np.concatenate(checking_balances)
savings_balances = np.concatenate(savings_balances)
credit_balances = np.concatenate(credit_balances)
meta = sum(meta, [])


#x_trends = []
#trends = []
#trend_days = 90
#for i in range(len(collapsed_total_balances)):
#    end = collapsed_total_balances[i][0]
#    start = end - datetime.timedelta(days=trend_days)
#    xs = []
#    ys = []
#    for j in range(i+1):
#        if start <= collapsed_total_balances[j][0] <= end:
#            xs.append(year_fraction(collapsed_total_balances[j][0]))
#            ys.append(collapsed_total_balances[j][1])
##    xs = np.array(xs) - xs[0]
#    extrapolated_annual_savings, _ = np.polyfit(xs, ys, 1)
#    x_trends.append(xs[-1])
#    trends.append(extrapolated_annual_savings)
    
x_trends = []
trends = []
trend_days = 90
end = collapsed_total_balances[0][0]
while end < collapsed_total_balances[-1][0]:
    end = end + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=trend_days)
    xs = []
    ys = []
    for j in range(len(collapsed_total_balances)):
        if start <= collapsed_total_balances[j][0] <= end:
            xs.append(year_fraction(collapsed_total_balances[j][0]))
            ys.append(collapsed_total_balances[j][1])
#    xs = np.array(xs) - xs[0]
    extrapolated_annual_savings, _ = np.polyfit(xs, ys, 1)
    x_trends.append(xs[-1])
    trends.append(extrapolated_annual_savings)
    

yellow_threshold = 100

def calculate(total_balances, checking_balances, savings_balances, credit_balances):
    x_data, total_balances_data = zip(*map(lambda (date, bal): (year_fraction(date), bal), total_balances))
    x_data = np.array(x_data)
    total_balances_data = np.array(total_balances_data)
    checking_balances_data = np.array(map(lambda (date, bal): bal, checking_balances))
    savings_balances_data = np.array(map(lambda (date, bal): bal, savings_balances))
    credit_balances_data = np.array(map(lambda (date, bal): bal, credit_balances))
    bank_balances_data = np.array(total_balances_data) - credit_balances_data

    total_balance_diffs = [total_balances_data[0]] + \
                          [total_balances_data[i] - total_balances_data[i-1]
                           for i in range(1, len(total_balances_data))]

    bank_balance_diffs = [bank_balances_data[0]] + \
                                         [bank_balances_data[i] - bank_balances_data[i-1]
                                          for i in range(1, len(bank_balances_data))]

    credit_balance_diffs = [credit_balances_data[0]] + \
                           [credit_balances_data[i] - credit_balances_data[i-1]
                            for i in range(1, len(credit_balances_data))]

    bank_rises = np.array([i for i in range(len(bank_balance_diffs)) if bank_balance_diffs[i] > 0])
    bank_small_falls = np.array([i for i in range(len(bank_balance_diffs)) \
                                 if bank_balance_diffs[i] < 0 and bank_balance_diffs[i] > -yellow_threshold])
    bank_large_falls = np.array([i for i in range(len(bank_balance_diffs)) \
                                 if bank_balance_diffs[i] <= -yellow_threshold])
    
    return (x_data, total_balances_data, checking_balances_data, savings_balances_data, credit_balances_data,
            bank_balances_data, total_balance_diffs, bank_balance_diffs, credit_balance_diffs,
            bank_rises, bank_small_falls, bank_large_falls)


calculated = calculate(total_balances, checking_balances, savings_balances, credit_balances)
collapsed_calculated = calculate(collapsed_total_balances, collapsed_checking_balances,
                                 collapsed_savings_balances, collapsed_credit_balances)

(x_data, total_balances_data, checking_balances_data, savings_balances_data, credit_balances_data,
 bank_balances_data, total_balance_diffs, bank_balance_diffs, credit_balance_diffs,
 bank_rises, bank_small_falls, bank_large_falls) = collapsed_calculated if table_collapse_days else calculated


##### Table output

yellow_threshold_color_rule = lambda x: 'red' if float(x[2]) <= -yellow_threshold else \
                                    ('yellow' if float(x[2]) < 0 else \
                                    ('white' if float(x[2]) == 0 else 'green'))
layout = [
    dict(name='Date', width=10, align='center'),
#    dict(name='Date', width=11, align='center'),
    dict(name='Type', width=8, align='right'),
    dict(name='$ Delta', width=11, superprefix='$', align='right', color_fn=yellow_threshold_color_rule),
    dict(name='% Delta', width=8, suffix='%', align='right', color_fn=yellow_threshold_color_rule),
    dict(name='Balance', width=11, superprefix='$', align='right', color='cyan'),
    dict(name='Checking', width=10, superprefix='$', align='right', color='yellow'),
    dict(name='Savings', width=11, superprefix='$', align='right', color='green'),
] + ([
    dict(name='Credit', width=10, superprefix='$', align='right', color='red'),
] if include_credit else []) + [
    dict(name='% Max', width=7, suffix='%', align='right'),
    dict(name='% Curr', width=7, suffix='%', align='right'),
]
    
table = DynamicConsoleTable(layout)
if not output_table_dividers:
    table.print_header()

total_balances = collapsed_total_balances if table_collapse_days else total_balances
meta = collapsed_meta if table_collapse_days else meta
current_balance = total_balances[-1][1]
max_balance = max(map(lambda x: x[1], total_balances))
last_month = None
last_year = None
last_date = None
for i in range(len(total_balances)):
    (date, balance) = total_balances[i]
    current_month = date.year*12 + date.month
    current_year = date.year
    new_month = current_month != last_month
    new_year = current_year != last_year
    last_month = current_month
    last_year = current_year
    
#    if i > 0:
#        table.finalize(heavy=new_month and output_table_dividers, divider=last_date != date)
#    if new_year and output_table_dividers:
#        table.print_divider(heavy=True)
    if i > 0:
        table.finalize(divider=last_date != date and not (new_month and output_table_dividers))
    if new_month and output_table_dividers:
        table.print_message(months_full[date.month-1] + ' ' + str(date.year), heavy=True)
        table.print_header()
        
    previous_balance = total_balances[i-1][1] if i > 0 else 0
    checking_balance = checking_balances[i][1]
    savings_balance = savings_balances[i][1]
    credit_balance = credit_balances[i][1]
    increased = total_balance_diffs[i] >= 0
    args = [
        date.strftime("%m/%d/%Y") if last_date != date else '',
#        months[date.month-1] + ' ' + str(date.day) + ' ' + str(date.year) if last_date != date else '',
        meta[i-1][1][0],
        ('+' if increased else '') + format_balance(total_balance_diffs[i]),
        ' --- ' if previous_balance == 0 else \
            ('+' if increased else '') + format_percent((balance - previous_balance) * 100 / previous_balance),
        format_balance(balance),
        format_balance(checking_balance),
        format_balance(savings_balance),
    ] + ([
        format_balance(abs(credit_balance)),
    ] if include_credit else []) + [
        format_percent(balance * 100 / max_balance),
        format_percent(balance * 100 / current_balance),
    ]
    table.update(*args)
    last_date = date
table.finalize(heavy=False, divider=True)
#table.print_header()

print
print 'Most recent '+str(trend_days)+'-day savings trend:', '$' + format_balance(trends[-1]/12) + ' per month,', \
                                                            '$' + format_balance(trends[-1]) + ' per year'


(x_data, total_balances_data, checking_balances_data, savings_balances_data, credit_balances_data,
 bank_balances_data, total_balance_diffs, bank_balance_diffs, credit_balance_diffs,
 bank_rises, bank_small_falls, bank_large_falls) = collapsed_calculated if plot_collapse_days else calculated


##### Plot output

x_smooth = np.linspace(min(x_data), max(x_data), 1000)
bank_balances_smooth = spline(x_data, bank_balances_data, x_smooth)
total_balances_smooth = spline(x_data, total_balances_data, x_smooth)
checking_balances_smooth = spline(x_data, checking_balances_data, x_smooth)
savings_balances_smooth = spline(x_data, savings_balances_data, x_smooth)
credit_balances_smooth = spline(x_data, credit_balances_data, x_smooth)

xs = x_data
bank_balances_ys = bank_balances_data
total_balances_ys = total_balances_data
checking_balances_ys = checking_balances_data
savings_balances_ys = savings_balances_data
credit_balances_ys = credit_balances_data

smooth = False

if smooth:
    xs = x_smooth
    bank_balances_ys = bank_balances_smooth
    total_balances_ys = bank_balances_smooth
    checking_balances_ys = checking_balances_smooth
    savings_balances_ys = savings_balances_smooth
    credit_balances_ys = credit_balances_smooth

total_balances_ys = np.array(total_balances_ys)
checking_balances_ys = np.array(checking_balances_ys)
savings_balances_ys = np.array(savings_balances_ys)
credit_balances_ys = np.abs(credit_balances_ys)

fig = plt.figure(figsize=(14,6))
plt.subplots_adjust(bottom=0.2, top=0.9)
plt.title('Balance vs. Time')
plt.xlabel('Time')
plt.ylabel('Balance ($)')

plt.fill_between(xs, savings_balances_ys, bank_balances_ys, facecolor='#3cb6e4', interpolate=True,
                 alpha=1.0, label='Checking')

plt.plot(xs, savings_balances_ys, color='black', linewidth=1, linestyle='--')
plt.fill_between(xs, [0] * len(xs), savings_balances_ys, facecolor='#9aca27', interpolate=True,
                 alpha=1.0, label='Savings')

plt.plot(xs, bank_balances_ys, color='black', linewidth=2.5, label='Checking + Savings Balance')

if include_credit:
    plt.fill_between(xs, total_balances_ys, bank_balances_ys, facecolor='#fa4747', interpolate=True, alpha=0.75)
    #plt.plot(xs, total_balances_ys, color='red', linewidth=1, linestyle=':')
    plt.plot(xs, credit_balances_ys, color='#ca3737', linewidth=2.5, label='Credit Balance')
    plt.fill_between(xs, [0] * len(xs), credit_balances_ys, facecolor='#fa4747', interpolate=True,
                     alpha=1.0, label='Credit')

plt.plot(xs, [0] * len(xs), color='black', linewidth=2)

scatter = plt.scatter(x_data[bank_rises], bank_balances_ys[bank_rises],
                      color='black', marker='o', facecolor='', edgecolor='green',
                      label='Incoming Transfers')
scatter = plt.scatter(x_data[bank_small_falls], bank_balances_ys[bank_small_falls],
                      color='black', marker='o', facecolor='', edgecolor='orange',
                      label='Outgoing Transfers (<$' + str(yellow_threshold) + ')')
scatter = plt.scatter(x_data[bank_large_falls], bank_balances_ys[bank_large_falls],
                      color='black', marker='o', facecolor='', edgecolor='red',
                      label='Outgoing Transfers ($' + str(yellow_threshold) + '+)')

xmin, xmax, ymin, ymax = plt.axis()
xticks = np.arange(int(min(x_data)), int(math.ceil(max(x_data)))+1./12, 1./12)
for x in xticks:
    plt.plot([x, x], [-1e9, 1e9], color='gray',
             linewidth=(1.5 if round(x / 0.01) * 0.01 % 0.5 == 0 else 0.3),
             alpha=(0.5 if round(x / 0.01) * 0.01 % 0.5 == 0 else 0.3))
regions = [(2015+5./12, 2015+8./12), (2016, 2016+1./12), (2016+5./12, 2016+8./12),
           (2017, 2017+1./12), (2017+5./12, 2017+8./12), (2018, 2018+1./12)]
for region in regions:
    plt.fill_between(region, [0, 0], [1e9, 1e9], facecolor='purple', alpha=0.075, interpolate=True)
    
plt.xticks(xticks, rotation=-60, ha='left')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.gca().set_xticklabels(transform_ticks(xticks))
plt.gca().grid(alpha=0.3)
plt.legend()


    
fig = plt.figure(figsize=(14,6))
plt.subplots_adjust(bottom=0.2, top=0.9)
plt.title('Trends over Time')
plt.xlabel('Time')
plt.ylabel('Extrapolated Annual Savings, ' + str(trend_days) + '-Day Trend ($)')

plt.plot(x_trends, trends)
plt.plot(x_trends, [0] * len(x_trends), color='black', linewidth=2)

xmin, xmax, ymin, ymax = plt.axis()
xticks = np.arange(int(min(x_trends)), int(math.ceil(max(x_trends)))+1./12, 1./12)
for x in xticks:
    plt.plot([x, x], [-1e9, 1e9], color='gray',
             linewidth=(1.5 if round(x / 0.01) * 0.01 % 0.5 == 0 else 0.3),
             alpha=(0.5 if round(x / 0.01) * 0.01 % 0.5 == 0 else 0.3))
for region in regions:
    plt.fill_between(region, [-1e9, -1e9], [1e9, 1e9], facecolor='purple', alpha=0.075, interpolate=True)
zeros = np.zeros(len(x_trends))
plt.fill_between(x_trends, trends, where=zeros<=trends, facecolor='green', alpha=0.5, interpolate=True)
plt.fill_between(x_trends, trends, where=zeros>=trends, facecolor='red', alpha=0.5, interpolate=True)
    
plt.xticks(xticks, rotation=-60, ha='left')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.gca().set_xticklabels(transform_ticks(xticks))
plt.gca().grid(alpha=0.3)

plt.show()

print
