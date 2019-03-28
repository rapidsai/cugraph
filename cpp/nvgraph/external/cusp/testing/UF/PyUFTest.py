import PyUFGet as uf
import shutil

def get_all_by_shape(rmin=50, rmax=1000, cmin=200, cmax=5000):
    matrices = uf.search(rowbounds = (rmin, rmax), colbounds=(cmin, cmax))

    print "Found {} matrices within the range ({}-{}) rows and ({}-{}) cols".format(len(matrices), rmin, rmax, cmin, cmax)
    for matrix in matrices:
        matrix.download(destpath='./matrices', extract=True)

        directory = './matrices/{}'.format(matrix.name)
        filename  = '{}.mtx'.format(matrix.name)
        shutil.move(directory + '/' + filename, directory + '.mtx')
        shutil.rmtree(directory)

if __name__ == "__main__":
    get_all_by_shape()

