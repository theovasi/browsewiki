import joblib

path_dict = joblib.load('data/filepath_dict.txt')

def tmp_gen():
    for i in range(len(path_dict)):
        path = path_dict[i]
        with open(path) as document_file:
            yield document_file.read()
