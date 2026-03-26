"""
DO NOT MODIFY - platform test runner.
The QCentroid platform replaces this file at runtime.
Use this file only for local testing: python app.py
"""
input_file_name = "input.json"
import json
with open(input_file_name) as f:
    dic = json.load(f)
extra_arguments = dic.get('extra_arguments', {})
solver_params = dic.get('solver_params', {})
import qcentroid
result = qcentroid.run(dic['data'], solver_params, extra_arguments)
print(result)
