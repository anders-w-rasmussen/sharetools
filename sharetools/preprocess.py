import numpy as np
import scipy.sparse
import pickle
from sklearn.preprocessing import normalize
from multiprocessing import Process, Manager
import gc
from halo import Halo
from io import StringIO
from skbio.stats.composition import ilr

from gneiss.cluster import random_linkage
from gneiss.balances import sparse_balance_basis
from scipy.special import softmax

from alive_progress import alive_bar, config_handler

###########################

# FUNCTIONS

def make_sparse_chunk(number_cells, number_peaks, txt_array, chunks):
    X = scipy.sparse.lil_matrix((number_cells, number_peaks))
    X[txt_array[:, 1] - 1, txt_array[:, 0] - 1] = txt_array[:, 2]
    chunks.append(X)

###########################

def run_preprocess(RNA_count_file, ATAC_count_file, ATAC_barcode_file,
                ATAC_peak_file, GTF_file, max_threads=2, velocity_file=None, outname=None, peak_per=None, cell_per=None, normalization='default', skip=False):

    # Find common barcodes and write to a list

    # spinner = Halo(text='Organizing Cell Barcodes', spinner='dots',
    #                color='white', placement='right')
    # spinner.start()

    skip_atac = skip

    assert normalization in ['default', 'ilr', 'none']

    s = open(RNA_count_file).readline().replace(' ', '\t')
    RNA_barcodes = np.loadtxt(StringIO(s), dtype=str)[1:]
    ATAC_barcodes = np.loadtxt(ATAC_barcode_file, dtype=str)
    ATAC_peaks = np.loadtxt(ATAC_peak_file, dtype=str)

    spinner = Halo(text='Organizing Cell Barcodes', spinner='dots',
                   color='white', placement='right')
    spinner.start()

    #THIS IS WHERE THAT CORRECTION IS
    for j in range(0, np.size(ATAC_barcodes)):
        ATAC_barcodes[j] = ATAC_barcodes[j].replace('.R', ',R').replace('.P', ',P')

    # OR
    # for j in range(0, np.size(ATAC_barcodes)):
    #     ATAC_barcodes[j] = ATAC_barcodes[j][:-5]
    # for j in range(0, np.size(RNA_barcodes)):
    #     RNA_barcodes[j] = RNA_barcodes[j][:-5]

    intersecting, rna_idx, atac_idx = np.intersect1d(RNA_barcodes, ATAC_barcodes, return_indices=True)
    common_idxs = np.hstack((rna_idx.reshape(-1, 1), atac_idx.reshape(-1, 1)))

    spinner.stop()
    print("Found Intersection of Barcodes")

    if skip_atac != False:

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

            # Filter on depths and peak coverage if applicable
            # CELLS x PEAKS

            non_zero = ATAC_counts.nonzero()

            depths = np.sum(ATAC_counts, axis=1)

            if cell_per != None:
                bad_depths = np.argwhere(depths <= np.percentile(depths, cell_per))
            else:
                bad_depths = np.argwhere(depths == 0)

            bad_idx_in_common = np.intersect1d(bad_depths, common_idxs)
            common_idxs = np.delete(common_idxs, (bad_idx_in_common), axis=0)

            # print("shape of ATAC counts matrix")
            # print(np.shape(ATAC_counts))
            # print("Nonzero elements")
            # print(np.shape(non_zero))


            if peak_per != None:
                peak_coverage = np.count_nonzero(ATAC_counts, axis=0)
                good_peaks = np.argwhere(peak_coverage >= np.percentile(peak_coverage, peak_per))
                bad_peaks = np.argwhere(peak_coverage < np.percentile(peak_coverage, peak_per))
                rem = np.sum(ATAC_counts[:, bad_peaks], axis=1)

                # COUNTS, BARCODES, PEAKS, DEPTHS
                if normalization == 'default':
                    pickle.dump(
                        [normalize(scipy.sparse.hstack([ATAC_counts[common_idxs[:, 1], good_peaks], rem]), norm='l1',
                                   axis=1), ATAC_barcodes[common_idxs[:, 1]],
                         np.concatenate((ATAC_peaks[good_peaks], np.asarray(['rem']))), depths[common_idxs[:, 1]]],
                        open("ATAC_data.p", "wb"))
                    print("Wrote Preprocessed ATAC-seq Data")

                if normalization == 'ilr':
                    pickle.dump(
                        [ilr(scipy.sparse.hstack([ATAC_counts[common_idxs[:, 1], good_peaks], rem]) + 1), ATAC_barcodes[common_idxs[:, 1]],
                         np.concatenate((ATAC_peaks[good_peaks], np.asarray(['rem']))), depths[common_idxs[:, 1]]],
                        open("ATAC_data.p", "wb"))
                    print("Wrote Preprocessed ATAC-seq Data")

                if normalization == 'none':
                    pickle.dump(
                        [scipy.sparse.hstack([ATAC_counts[common_idxs[:, 1], good_peaks], rem]), ATAC_barcodes[common_idxs[:, 1]],
                         np.concatenate((ATAC_peaks[good_peaks], np.asarray(['rem']))), depths[common_idxs[:, 1]]],
                        open("ATAC_data.p", "wb"))
                    print("Wrote Preprocessed ATAC-seq Data")

                del ATAC_counts
                gc.collect()

            else:

                if normalization == 'default':
                    # COUNTS, BARCODES, PEAKS, DEPTHS
                    pickle.dump(
                        [normalize(ATAC_counts[common_idxs[:, 1], :], norm='l1', axis=1), ATAC_barcodes[common_idxs[:, 1]],
                         ATAC_peaks, depths[common_idxs[:, 1]]], open("ATAC_data.p", "wb"))
                    print("Wrote Preprocessed ATAC-seq Data")

                if normalization == 'ilr':
                    # COUNTS, BARCODES, PEAKS, DEPTHS
                    pickle.dump(
                        [ilr(ATAC_counts[common_idxs[:, 1], :] + 1), ATAC_barcodes[common_idxs[:, 1]],
                         ATAC_peaks, depths[common_idxs[:, 1]]], open("ATAC_data.p", "wb"))
                    print("Wrote Preprocessed ATAC-seq Data")

                if normalization == 'none':
                    # COUNTS, BARCODES, PEAKS, DEPTHS
                    pickle.dump(
                        [ATAC_counts[common_idxs[:, 1], :], ATAC_barcodes[common_idxs[:, 1]],
                         ATAC_peaks, depths[common_idxs[:, 1]]], open("ATAC_data.p", "wb"))
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

    GTF_info = np.loadtxt(GTF_file, dtype=str, skiprows=5, delimiter='\t')

    rna_depths = np.sum(RNA_counts, axis=1)

    if normalization == 'default':
        pickle.dump([normalize(RNA_counts[common_idxs[:, 0], :], norm='l1', axis=1), RNA_barcodes[common_idxs[:, 0]], gene_names, rna_depths[common_idxs[:, 0]], GTF_info], open( "RNA_data.p", "wb" ))
    if normalization == 'ilr':
        pickle.dump([ilr(RNA_counts[common_idxs[:, 0], :] + 1), RNA_barcodes[common_idxs[:, 0]], gene_names, rna_depths[common_idxs[:, 0]], GTF_info], open( "RNA_data.p", "wb" ))
    if normalization == 'none':
        pickle.dump([RNA_counts[common_idxs[:, 0], :], RNA_barcodes[common_idxs[:, 0]], gene_names, rna_depths[common_idxs[:, 0]], GTF_info], open( "RNA_data.p", "wb" ))

    del RNA_counts
    gc.collect()

    spinner.stop()
    print("Wrote Preprocessed RNA-seq Data")
    print("Preprocessing Completed")
