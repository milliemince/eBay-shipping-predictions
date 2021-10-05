# ## Converts .tsv file into .csv file that can be used by pandas readcsv() function

import pandas as pd
import sys

#get .tsv file to convert
tsv_file = sys.argv[1]
tsv_filename = tsv_file[:-4]

#turn into csv file
csv_table = pd.read_table(tsv_file,sep='\t')
csv_filename = tsv_filename + ".csv"

#save csv file
csv_table.to_csv(csv_filename, index=False)
