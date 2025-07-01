from DataCleaner import DataCleaner

# 1. Create an instance of DataCleaner with your dataset path
cleaner = DataCleaner(data_file_path="/home/becode/Desktop/Python-Hajer/ImmoElizaML/ImmoEliZaML/data/immoweb-dataset.csv",
                      postcode_file_path="/home/becode/Desktop/Python-Hajer/ImmoElizaML/ImmoEliZaML/data/code-postaux-belge.csv")

cleaner.send_output_file("ImmoEliZaML/data/cleaned_data.csv")
