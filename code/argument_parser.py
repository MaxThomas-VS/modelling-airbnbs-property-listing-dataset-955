import datetime
import argparse

def make_args(parser=argparse.ArgumentParser()):
    parser.add_argument('--file_name', type=str, required=False, default='listing.csv')
    parser.add_argument('--import_path', type=str, required=False, default='../data/raw/airbnb-property-listings/tabular_data/')
    parser.add_argument('--export_path', type=str, required=False, default='../data/intermediate/airbnb-property-listings/tabular_data/')
    parser.add_argument('--ml_model', type=str, required=False, default='../data/intermediate/airbnb-property-listings/tabular_data/')
    args = vars(parser.parse_args())
    args['script'] = parser.prog
    args['run_time'] = datetime.datetime.now()
    return args
