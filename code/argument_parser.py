import datetime
import argparse

def make_args(parser=argparse.ArgumentParser()):
    parser.add_argument('--file_name', type=str, required=False, default='listing.csv')
    parser.add_argument('--import_path', type=str, required=False, default='../data/raw/airbnb-property-listings/tabular_data/')
    parser.add_argument('--export_path', type=str, required=False, default='../data/intermediate/airbnb-property-listings/tabular_data/')
    parser.add_argument('--ml_model', type=comma_split_str, required=False, default=['SGDRegressor'])
    parser.add_argument('--random_state', type=int, required=False, default=100)
    parser.add_argument('--nn_n_configs', type=int, required=False, default=16)
    parser.add_argument('--nn_epochs', type=int, required=False, default=50)
    args = vars(parser.parse_args())
    args['script'] = parser.prog
    args['run_time'] = datetime.datetime.now()
    return args

def comma_split_str(arg_in):
    return arg_in.split(',')

def comma_split_float(arg_in):
    return list(map(float, arg_in.split(',')))
