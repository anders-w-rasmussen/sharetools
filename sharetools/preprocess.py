import numpy as np
import scipy.sparse
import pickle
from sklearn.preprocessing import normalize
from multiprocessing import Process, Manager
import gc
from halo import Halo
from io import StringIO
from alive_progress import alive_bar, config_handler

###########################

# FUNCTIONS

def make_sparse_chunk(number_cells, number_peaks, txt_array, chunks):
    X = scipy.sparse.lil_matrix((number_cells, number_peaks))
    X[txt_array[:, 1] - 1, txt_array[:, 0] - 1] = txt_array[:, 2]
    chunks.append(X)

###########################

def run_preprocess(RNA_count_file, ATAC_count_file, ATAC_barcode_file,
                ATAC_peak_file, GTF_file, max_threads=2, velocity_file = None, outname = None):

    # Find common barcodes and write to a list

    # spinner = Halo(text='Organizing Cell Barcodes', spinner='dots',
    #                color='white', placement='right')
    # spinner.start()

    s = open(RNA_count_file).readline().replace(' ', '\t')
    RNA_barcodes = np.loadtxt(StringIO(s), dtype=str)[1:]
    ATAC_barcodes = np.loadtxt(ATAC_barcode_file, dtype=str)
    ATAC_peaks = np.loadtxt(ATAC_peak_file, dtype=str)

    spinner = Halo(text='Organizing Cell Barcodes', spinner='dots',
                   color='white', placement='right')
    spinner.start()


    #THIS IS WHERE THAT CORRECTION IS
    # for j in range(0, np.size(ATAC_barcodes)):
    #     ATAC_barcodes[j] = ATAC_barcodes[j].replace('.R', ',R').replace('.P', ',P')

    # OR
    for j in range(0, np.size(ATAC_barcodes)):
        ATAC_barcodes[j] = ATAC_barcodes[j][:-5]
    for j in range(0, np.size(RNA_barcodes)):
        RNA_barcodes[j] = RNA_barcodes[j][:-5]


    intersecting, rna_idx, atac_idx = np.intersect1d(RNA_barcodes, ATAC_barcodes, return_indices=True)
    common_idxs = np.hstack((rna_idx.reshape(-1, 1), atac_idx.reshape(-1, 1)))

    spinner.stop()
    print("Found Intersection of Barcodes")



    # Now let's get the ATAC-seq counts

    n_threads = max_threads

    spinner = Halo(text='Loading ATAC-seq counts into memory', spinner='dots',
                    color='white', placement='right')
    spinner.start()

    txt_array = np.asarray(np.loadtxt(ATAC_count_file, skiprows=1, dtype=str)[1:,:], dtype=int)

    spinner.stop()

    number_peaks = np.max(txt_array[:, 0])
    number_cells = np.max(txt_array[:, 1])

    # Split file across cores and write to a sparse matrix

    spinner = Halo(text='Normalizing ATAC-seq', spinner='dots',
                    color='white', placement='right')
    spinner.start()

    split_list = np.array_split(txt_array, n_threads)
    del txt_array
    gc.collect()

    ATAC_counts = scipy.sparse.lil_matrix((number_cells, number_peaks))

    #Looks like two cores is the fastest on my computer
    if __name__ != "__main__":
        with Manager() as manager:
            chunks = manager.list()
            processes = []
            for k in range(0, n_threads):

                p = Process(target=make_sparse_chunk, args=(number_cells, number_peaks, split_list[k], chunks))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            for chunk in chunks:
                nonzero = chunk.nonzero()
                ATAC_counts[nonzero] = chunk[nonzero]
                del chunk
                gc.collect()

        # Write to hard drive and remove from memory

        depths = np.sum(ATAC_counts, axis=1)

        # HACK

        bad_idx = np.argwhere(depths == 0)
        print(bad_idx)

        bad_idx_in_common = np.argwhere(common_idxs[:, 1])
        common_idxs = np.delete(common_idxs, (bad_idx_in_common), axis=0)


        # COUNTS, BARCODES, PEAKS, DEPTHS
        pickle.dump([normalize(ATAC_counts[common_idxs[:, 1], :], norm='l1', axis=1), ATAC_barcodes[common_idxs[:, 1]], ATAC_peaks, depths[common_idxs[:, 1]]], open( "ATAC_data.p", "wb" ) )
        print("Wrote Preprocessed ATAC-seq Data")

        del ATAC_counts
        gc.collect()

    spinner.stop()

    spinner = Halo(text='Normalizing RNA-seq', spinner='dots',
                    color='white', placement='right')
    spinner.start()

    # Load in the RNA-seq
    RNA_counts = scipy.sparse.lil_matrix(np.asarray(np.loadtxt(RNA_count_file, skiprows=1, dtype=str)[:,1:], dtype=int).transpose())
    gene_names = np.loadtxt(RNA_count_file, usecols=0,
                            skiprows=1, dtype=str)

    # COUNTS, BARCODES, GENE NAMES

    rna_depths = np.sum(RNA_counts, axis=1)

    pickle.dump([normalize(RNA_counts[common_idxs[:, 0], :], norm='l1', axis=1), RNA_barcodes[common_idxs[:, 0]], gene_names, rna_depths[common_idxs[:, 0]]], open( "RNA_data.p", "wb" ))

    del RNA_counts
    gc.collect()

    spinner.stop()
    print("Wrote Preprocessed RNA-seq Data")
    print("Preprocessing Completed")