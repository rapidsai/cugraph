#!/usr/bin/env python 
import os,csv

device_id = '0'  # index of the device to use

binary_filename = '../spmv'                  # command used to run the tests
output_file = 'benchmark_output.log'        # file where results are stored


# The unstructured matrices are available online:
#    http://www.nvidia.com/content/NV_Research/matrices.zip

mats = []
unstructured_path = '~/scratch/Matrices/williams/mm/'
unstructured_mats = [('Dense','dense2.mtx'),
                     ('Protein','pdb1HYS.mtx'),
                     ('FEM/Spheres','consph.mtx'),
                     ('FEM/Cantilever','cant.mtx'),
                     ('Wind Tunnel','pwtk.mtx'),
                     ('FEM/Harbor','rma10.mtx'),
                     ('QCD','qcd5_4.mtx'),
                     ('FEM/Ship','shipsec1.mtx'),
                     ('Economics','mac_econ_fwd500.mtx'),
                     ('Epidemiology','mc2depi.mtx'),    
                     ('FEM/Accelerator','cop20k_A.mtx'),
                     ('Circuit','scircuit.mtx'),
                     ('Webbase','webbase-1M.mtx'),
                     ('LP','rail4284.mtx') ]
unstructured_mats = [ mat + (unstructured_path,) for mat in unstructured_mats]

structured_path = '~/scratch/Matrices/stencil/'
structured_mats = [('Laplacian_3pt_stencil',  '3pt_1000000.mtx'),
                   ('Laplacian_5pt_stencil',  '5pt_1000x1000.mtx'),
                   ('Laplacian_7pt_stencil',  '7pt_100x100x100.mtx'),
                   ('Laplacian_9pt_stencil',  '9pt_1000x1000.mtx'),
                   ('Laplacian_27pt_stencil', '27pt_100x100x100.mtx')]
structured_mats = [ mat + (structured_path,) for mat in structured_mats]

# assemble suite of matrices
trials = unstructured_mats  + structured_mats


def run_tests(value_type):
    # remove previous result (if present)
    open(output_file,'w').close()
    
    # run benchmark for each file
    for matrix,filename,path in trials:
        matrix_filename = path + filename

        # setup the command to execute
        cmd = binary_filename 
        cmd += ' ' + matrix_filename                  # e.g. pwtk.mtx
        cmd += ' --device=' + device_id               # e.g. 0 or 1
        cmd += ' --value_type=' + value_type          # e.g. float or double

        # execute the benchmark on this file
        os.system(cmd)
    
    # process output_file
    matrices = {}
    results = {}
    kernels = set()
    #
    fid = open(output_file)
    for line in fid.readlines():
        tokens = dict( [tuple(part.split('=')) for part in line.split()] )
    
        if 'file' in tokens:
            file = os.path.split(tokens['file'])[1]
            matrices[file] = tokens
            results[file] = {}
        else:
            kernel = tokens['kernel']
            results[file][kernel] = tokens
            kernels.add(tokens['kernel'])
    
    ## put CPU results before GPU results
    #kernels = ['csr_serial'] + sorted(kernels - set(['csr_serial']))
    kernels = sorted(kernels)

    # write out CSV formatted results
    def write_csv(field):
        fid = open('bench_' + value_type + '_' + field + '.csv','w')
        writer = csv.writer(fid)
        writer.writerow(['matrix','file','rows','cols','nonzeros'] + kernels)
        
        for (matrix,file,path) in trials:
            line = [matrix, file, matrices[file]['rows'], matrices[file]['cols'], matrices[file]['nonzeros']]
        
            matrix_results = results[file]
            for kernel in kernels:
                if kernel in matrix_results:
                    line.append( matrix_results[kernel][field] )
                else:
                    line.append(' ')
            writer.writerow( line )
        fid.close()
    
    write_csv('gflops') #GFLOP/s
    write_csv('gbytes') #GBytes/s


run_tests('float')
run_tests('double')
 
