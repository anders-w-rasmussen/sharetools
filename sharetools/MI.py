import numpy as np
import pickle
import scipy.stats
import scipy.sparse
from sklearn.feature_selection import mutual_info_regression
import csv
from io import StringIO
from time import time
from halo import Halo
from scipy.stats.stats import pearsonr
import warnings
from sklearn.exceptions import DataConversionWarning
from alive_progress import alive_bar, config_handler
from scipy.special import hyp2f1



def mutual_information(ATAC_pickled, RNA_pickled, gene=None, chrom=None, start=None, stop=None, mode='expression', num_threads=1, window_size=30, cutoff = 10):


    # Haven't parallelized any of this yet

    if any([gene, start, stop]) != None:

        window_size_kb = window_size


        # Individual Mode
        assert all([gene, start, stop]) != None, 'gene, start, stop arguments must all be specified or none must be specified'

        # ATAC list is [COUNTS, BARCODES, PEAKS, DEPTHS]
        ATAC_list = pickle.load(open(ATAC_pickled, "rb"))

        # RNA list is [COUNTS, BARCODES, GENE NAMES]
        RNA_list = pickle.load(open(RNA_pickled, "rb"))

        # Find the index of the gene we specified
        gene_idx = np.argwhere(RNA_list[2] == gene)[0]

        # find the peaks that are within the specified range
        idxs = []
        peaks = []
        # Could vectorize this but fast anyway
        for i in range(0, np.size(ATAC_list[2], axis=0)):
            if ATAC_list[2][i, 0] == chrom:
                if int(ATAC_list[2][i, 1]) in range(start, stop):
                    idxs.append(i)
                    peaks.append([ATAC_list[2][i, 1], ATAC_list[2][i, 2]])
        idxs = np.asarray(idxs, dtype=int)

        # Sample some background peaks around the gene
        idxs_bkg = []
        # Could vectorize this but fast anyway
        for i in range(0, np.size(ATAC_list[2], axis=0)):
            if ATAC_list[2][i, 0] == chrom:
                if int(ATAC_list[2][i, 1]) in range(stop + 200, stop + window_size_kb * 1000):
                    idxs_bkg.append(i)

                if int(ATAC_list[2][i, 1]) in range(start - window_size_kb * 1000, start - 200):
                    idxs_bkg.append(i)


        idxs_bkg = np.asarray(idxs_bkg, dtype=int)

        print('         ')
        print("Number of Background Peaks to Consider")
        print(np.size(idxs_bkg))
        print('         ')


        spinner = Halo(text='Calculating Mutual Information and Rho for Peaks of Interest', spinner='dots',
                       color='white', placement='right')
        spinner.start()

        warnings.filterwarnings(action='ignore', category=DataConversionWarning)

        # nonzero gene counts cells
        nonzero_rna = np.argwhere(RNA_list[0][:, gene_idx].todense() != 0)

        MI = []
        rhos = []
        badguys = []
        for i in range(0, np.size(idxs)):

            # Find indices of cells that have nonzero counts in both
            nonzero_atac = np.argwhere(ATAC_list[0][:, idxs[i]].todense() != 0)
            inboth = np.intersect1d(nonzero_atac, nonzero_rna)

            if np.size(inboth) <= cutoff:
                # print("Too sparse a peak to consider")
                badguys.append(i)



            # Need to filter out anything has stddev = 0
            if np.std(np.asarray(ATAC_list[0][inboth, idxs[i]].todense().reshape(-1, 1), dtype=float)) == 0:
                MI.append(0)
                rhos.append(0)
                continue
            else:
                if np.std(np.asarray(RNA_list[0][inboth, gene_idx].reshape(-1, 1), dtype=float)) == 0:
                    MI.append(0)
                    rhos.append(0)
                    continue
            try:
                MI.append(mutual_info_regression(ATAC_list[0][inboth, idxs[i]].todense(), RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2, n_neighbors=3, discrete_features=False))
            except ValueError:
                MI.append(mutual_info_regression(ATAC_list[0][inboth, idxs[i]].todense(), RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2, n_neighbors=1, discrete_features=False))

            unadjusted_rho = np.corrcoef(ATAC_list[0][inboth, idxs[i]].todense().transpose(), RNA_list[0][inboth, gene_idx].reshape(-1, 1).transpose())[0, 1]
            n = np.size(ATAC_list[0][inboth, idxs_bkg[i]].todense().transpose())
            adjusted_rho = unadjusted_rho * hyp2f1(1 / 2, 1 / 2, (n - 1) / 2, 1 - unadjusted_rho ** 2)
            rhos.append(adjusted_rho)



        MI = np.asarray(MI)
        rhos = np.asarray(rhos)


        spinner.stop()

        max_idx = np.size(ATAC_list[2], axis=0)


        MI_bkg = []
        rhos_bkg = []
        bad_peaks = 0

        print("Calculating Mutual Information and Rho from Background")



        L = np.size(idxs_bkg)
        with alive_bar(L) as bar:


            for i in range(0, np.size(idxs_bkg)):

                nonzero_atac = np.argwhere(ATAC_list[0][:, idxs_bkg[i]].todense() != 0)
                inboth = np.intersect1d(nonzero_atac, nonzero_rna)


                if np.size(inboth) <=cutoff:
                    #print("Too sparse a peak to consider")
                    bad_peaks += 1

                    bar()
                    continue

                else:

                    # Need to filter out anything has stddev = 0
                    if np.std(np.asarray(ATAC_list[0][inboth, idxs_bkg[i]].todense().reshape(-1, 1), dtype=float)) == 0:
                        MI_bkg.append(0)
                        rhos_bkg.append(0)
                        bar()
                        continue
                    else:
                        if np.std(np.asarray(RNA_list[0][inboth, gene_idx].reshape(-1, 1), dtype=float)) == 0:
                            MI_bkg.append(0)
                            rhos_bkg.append(0)
                            bar()
                            continue

                    try:

                        MI_bkg.append(mutual_info_regression(ATAC_list[0][inboth, idxs_bkg[i]].todense(),
                                                         RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2,
                                                         n_neighbors=3, discrete_features=False))
                    except:

                        print("A Mutual Information Calculation had n_samples < n_neighbors=3. Switched nn to 1.")
                        MI_bkg.append(mutual_info_regression(ATAC_list[0][inboth, idxs_bkg[i]].todense(),
                                                             RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2,
                                                             n_neighbors=1, discrete_features=False))


                    unadjusted_rho = np.corrcoef(ATAC_list[0][inboth, idxs_bkg[i]].todense().transpose(),
                                            RNA_list[0][inboth, gene_idx].reshape(-1, 1).transpose())[0, 1]

                    n = np.size(ATAC_list[0][inboth, idxs_bkg[i]].todense().transpose())
                    adjusted_rho = unadjusted_rho * hyp2f1(1/2, 1/2, (n-1)/2, 1-unadjusted_rho**2)
                    rhos_bkg.append(adjusted_rho)

                    bar()

        MI_bkg = np.asarray(MI_bkg)
        rhos_bkg = np.asarray(rhos_bkg)

        print(str(bad_peaks) + ' peaks were too sparse to include in background distribution')

        # Using Bonferroni Correction here
        n_peaks = np.size(MI)
        cutoff = float(5)/float(n_peaks)
        sig_lvl = float(cutoff) / float(100)

        high_percentile = np.percentile(MI_bkg, 100 - cutoff)
        high_percentile_corr = np.percentile(rhos_bkg, 100 - cutoff)


        print('         ')
        print("Bonferroni Adjusted Percentile for Significance:")
        print(100 - cutoff)
        print('         ')
        print("Number of Background Peaks Sampled")
        print(np.size(idxs_bkg) - bad_peaks)
        print('         ')

        print("Value for Significance MI:")
        print(high_percentile)
        print('         ')
        print("Value for Significance Rho:")
        print(high_percentile_corr)
        print('         ')




        txt_outs = []
        txt_corr_out = []
        for i in range(0, n_peaks):
            if MI[i] >= high_percentile:
                txt_outs.append('Yes')
            else:
                txt_outs.append('  ')
            if rhos[i] >= high_percentile_corr:
                txt_corr_out.append('Yes')
            else:
                txt_corr_out.append(' ')

        print('Gene Name: ' + gene)
        print(' ')


        print('Chromosome:' + ' ' + str(chrom))
        print(' ')

        print("Peak Start         Peak Stop      Mutual Information     Significant MI        Correlation        Significant Correlation (positive)")
        print( ' ')


        for i in range(0, n_peaks):

            if i in badguys:
                print(str(peaks[i][0]) + '           ' + str(peaks[i][1]) + '       ' + str(MI[i][0]) + '          ' + 'too sparse' + '                ' + str(rhos[i]) + '         ' + 'too sparse ')
            else:
                print(str(peaks[i][0]) + '           ' + str(peaks[i][1]) + '       ' + str(MI[i][0]) + '          ' +  txt_outs[i] + '                ' + str(rhos[i]) + '         ' + txt_corr_out[i])
            