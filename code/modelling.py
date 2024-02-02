import argument_parser as ap
from tabular_data import ModelData

args = ap.make_args()
print(args)

model_data = ModelData()

df = model_data.load_tabular_data(args['file_name'], args['import_path'])