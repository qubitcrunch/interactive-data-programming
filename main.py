from qubitcrunch.core import *
import inspect
print(labeling_functions_return())
try:
    weakly_label(weak="snorkel")
except:
    print("weak labeling errored!")

def lf_new_science_term(x):
    if type(x) == pd.core.series.Series:
        x = x.body
    if type(x) != str:
        x = str(x)

    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(x)

    automobile_terms = ["atom"]
    for token in tokens:
        if token in automobile_terms:
            return 3
        else:
            return -1

labeling_function_str=inspect.getsource(lf_new_science_term)
labeling_function_new(labeling_function_str)
importlib.import_module("qubitcrunch.labeling_functions")
all_lfs = [obj for name, obj in inspect.getmembers(sys.modules["qubitcrunch.labeling_functions"]) if isinstance(obj, snorkel.labeling.lf.core.LabelingFunction)]
print(all_lfs)





