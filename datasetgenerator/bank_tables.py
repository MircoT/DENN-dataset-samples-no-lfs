from collections import namedtuple

# function map
__all__ = ['get_bank_map_attributes']

# 2-job
job_table = [
    'admin.',
    'blue-collar',
    'entrepreneur',
    'housemaid',
    'management',
    'retired',
    'self-employed',
    'services',
    'student',
    'technician',
    'unemployed',
    'unknown'
]
# 3-material
marital_table = [
    'divorced',
    'married',
    'single',
    'unknown'
]
# 4-education
education_table = [
    'basic.4y',
    'basic.6y',
    'basic.9y',
    'high.school',
    'illiterate',
    'professional.course',
    'university.degree',
    'unknown'
]
# 5-default
# 6-housing
# 7-loan
default_housing_loan_table = [
    'no',
    'unknown',
    'yes'
]
# 8 - contact
contact_table = [
    'cellular',
    'telephone'
]
# 9-month
month_table = [
    'jan',
    'feb',
    # added by me,
    'mar',
    'apr',
    'may',
    'jun',
    'jul',
    'aug',
    'sep',
    'oct',
    # ,
    'nov',
    'dec'
]
# 10 - day_of_week
day_of_week_table = [
    'mon',
    'tue',
    'wed',
    'thu',
    'fri'
]
# 15-poutcome
poutcome_table = [
    'failure', 'nonexistent', 'success'
]


def create_map_from_table_0to1(table):
    c_factor = 1.0 / (len(table)-1)
    t_out = {}
    for i, name in enumerate(table):
        t_out[name] = float(i) * c_factor
    return t_out


def create_map_from_table_buckets_middle(table):
    b_size = 1.0 / len(table)
    b_offset = b_size / 2.0
    t_out = {}
    for i, name in enumerate(table):
        t_out[name] = b_size*float(i) + b_offset
    return t_out

tables_table = [
    (2, 'job', job_table, create_map_from_table_buckets_middle),
    (3, 'marital', marital_table, create_map_from_table_buckets_middle),
    (4, 'education', education_table, create_map_from_table_buckets_middle),
    (5, 'default', default_housing_loan_table,
     create_map_from_table_buckets_middle),
    (6, 'housing', default_housing_loan_table,
     create_map_from_table_buckets_middle),
    (7, 'loan', default_housing_loan_table,
     create_map_from_table_buckets_middle),
    (8, 'contact', contact_table, create_map_from_table_buckets_middle),
    (9, 'month', month_table, create_map_from_table_buckets_middle),
    (10, 'day_of_week', day_of_week_table,
     create_map_from_table_buckets_middle),
    (15, 'poutcome', poutcome_table, create_map_from_table_buckets_middle)
]


def create_maps_from_tables(tables):
    id_map = {}
    name_map = {}
    for (id_, name, table, fun) in tables:
        values = fun(table)
        id_map[id_] = values
        name_map[name] = values
    return id_map, name_map


def get_bank_map_attributes():
    return create_maps_from_tables(tables_table)
