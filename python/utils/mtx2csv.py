import os
import time
from scipy.io import mmread
import argparse


parser = argparse.ArgumentParser(description='Convert the sparsity pattern \
                                 of a MatrixMarket file into a CSV file. \
                                 Each directed edge is explicitely stored, \
                                 edges are unsorted, IDs are 0-based.')
parser.add_argument('file', type=argparse.FileType(),
                    help='Path to the MatrixMarket file')
parser.add_argument('--csv_separator_name', type=str, default="space",
                    choices=["space", "tab", "comma"],
                    help='csv separator can be : \
                    space, tab or comma. Default is space')
args = parser.parse_args()

# Read
print('Reading ' + str(args.file.name) + '...')
t1 = time.time()
M = mmread(args.file.name).asfptype()
read_time = time.time()-t1
print('Time (s) : ' + str(round(read_time, 3)))

print('V ='+str(M.shape[0])+', E = '+str(M.nnz))

if args.csv_separator_name == "space":
    separator = ' '
elif args.csv_separator_name == "tab":
    separator = '	'
elif args.csv_separator_name == "comma":
    separator = ','
else:
    parser.error("supported csv_separator_name values are space, tab, comma")

# Write
print('Writing CSV file: '
      + os.path.splitext(os.path.basename(args.file.name))[0] + '.csv ...')
t1 = time.time()
os.path.splitext(os.path.basename(args.file.name))[0] + '.csv'
csv_file = open(os.path.splitext(os.path.basename(args.file.name))[0]
                + '.csv', "w")
for item in range(M.getnnz()):
    csv_file.write("{}{}{}\n".format(M.row[item], separator, M.col[item]))
csv_file.close()
write_time = time.time()-t1
print('Time (s) : ' + str(round(write_time, 3)))
