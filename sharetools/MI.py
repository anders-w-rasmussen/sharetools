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
from multiprocessing import Process, Manager
from sklearn.exceptions import DataConversionWarning
from sklearn.mixture import GaussianMixture
from alive_progress import alive_bar, config_handler
from scipy.special import hyp2f1
from info_theory_fns import entropy_decomp_3
from matplotlib import pyplot as plt


def mutual_information(ATAC_pickled, RNA_pickled, gene=None, chrom=None, start=None, stop=None, mode='expression', num_threads=1, window_size=30, cutoff=10):

    # COUPLE ISSUES HERE:
    # i) haven't parallelized this yet
    # ii) I'm taking a window around the gene, is this legit?
    # iii) I'm not correcting the Mutual Information score, does it need to be adjusted?

    cutoff = 10

    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    if gene != None:

        print("Looking at one gene: " + gene)

        window_size_kb = window_size

        # Individual Mode
        assert all([gene, start, stop]) != None, 'gene, start, stop arguments must all be specified or none must be specified'

        # ATAC list is [COUNTS, BARCODES, PEAKS, DEPTHS]
        ATAC_list = pickle.load(open(ATAC_pickled, "rb"))

        # RNA list is [COUNTS, BARCODES, GENE NAMES]
        RNA_list = pickle.load(open(RNA_pickled, "rb"))

        # Find the index of the gene we specified

        assert np.size(np.argwhere(RNA_list[2] == gene)) != 0, "The gene you specified is not in this dataset"
        gene_idx = np.argwhere(RNA_list[2] == gene)[0]

        # Heres the density switch
        RNA_list[0] = RNA_list[0].todense()



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

        max_idx = np.size(ATAC_list[2], axis=0)

        MI_bkg = []
        rhos_bkg = []
        bad_peaks = 0

        nonzero_rna = np.argwhere(RNA_list[0][:, gene_idx] != 0)


        print("Creating Background Distribution of MIs and Rhos")

        L = np.size(idxs_bkg)
        with alive_bar(L) as bar:

            for i in range(0, np.size(idxs_bkg)):

                nonzero_atac = np.argwhere(ATAC_list[0][:, idxs_bkg[i]].todense() != 0)
                inboth = np.intersect1d(nonzero_atac, nonzero_rna)

                if np.size(inboth) <= cutoff:
                    # print("Too sparse a peak to consider")
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

                    MI_bkg.append(mutual_info_regression(ATAC_list[0][inboth, idxs_bkg[i]].todense(),
                                                             RNA_list[0][inboth, gene_idx].reshape(-1, 1),
                                                             random_state=2,
                                                             n_neighbors=3, discrete_features=False))
                    # except:
                    #
                    #     print(np.shape(ATAC_list[0][inboth, idxs_bkg[i]].todense()))
                    #     print(np.shape(RNA_list[0][inboth, gene_idx].reshape(-1, 1)))
                    #
                    #     print("Too few reads in this peak. skipping it. ")
                        # print("A Mutual Information Calculation had n_samples < n_neighbors=3. Switched nn to 1.")
                        # MI_bkg.append(mutual_info_regression(ATAC_list[0][inboth, idxs_bkg[i]].todense(),
                        #                                      RNA_list[0][inboth, gene_idx].reshape(-1, 1),
                        #                                      random_state=2,
                        #                                      n_neighbors=1, discrete_features=False))


                    # print(np.shape( RNA_list[0][inboth, gene_idx].todense()))
                    # print(np.shape(ATAC_list[0][inboth, idxs_bkg[i]].todense()))
                    unadjusted_rho = np.corrcoef(ATAC_list[0][inboth, idxs_bkg[i]].todense().transpose(),
                                                 RNA_list[0][inboth, gene_idx].reshape(-1, 1).transpose())[0, 1]

                    # print(unadjusted_rho)

                    n = np.size(ATAC_list[0][inboth, idxs_bkg[i]].todense().transpose())
                    adjusted_rho = unadjusted_rho * hyp2f1(1 / 2, 1 / 2, (n - 1) / 2, 1 - unadjusted_rho ** 2)

                    if np.isnan(adjusted_rho) == True:
                        rhos_bkg.append(0)
                    else:
                        rhos_bkg.append(unadjusted_rho)

                    bar()

        MI_bkg = np.asarray(MI_bkg)
        rhos_bkg = np.asarray(rhos_bkg)

        # Check to see if the background distribution of Rhos is Gaussian mixture

        g_1 = GaussianMixture(n_components=1, random_state=0).fit(rhos_bkg.reshape(-1, 1))
        g_2 = GaussianMixture(n_components=2, random_state=0).fit(rhos_bkg.reshape(-1, 1))

        BIC_1_bkg = g_1.bic(rhos_bkg.reshape(-1, 1))
        BIC_2_bkg = g_2.bic(rhos_bkg.reshape(-1, 1))

        if BIC_1_bkg <= BIC_2_bkg:
            print("Distribution of background correlation appears unimodal (Bayesian Info Criterion " + str(BIC_1_bkg) + " for gaussian vs. " + str(
                BIC_2_bkg) + " for mixture)")
        elif BIC_2_bkg < BIC_1_bkg:
            print("Distribution of background correlation appears bimodal (Bayesian Info Criterion " + str(BIC_1_bkg) + " for gaussian vs. " + str(
                BIC_2_bkg) + " for mixture)")

        print(str(bad_peaks) + ' peaks were too sparse to include in background distribution')

        # nonzero gene counts cells

        MI = []
        rhos = []
        badguys = []

        print("Calculating Mutual Information and Rho for Peaks in Specified Region")

        with alive_bar(np.size(idxs)) as bar:
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
                    try:
                        if np.std(np.asarray(RNA_list[0][inboth, gene_idx], dtype=float)) == 0:
                            MI.append(0)
                            rhos.append(0)
                            continue
                    except:
                        #print("Something went wrong here")
                        MI.append(0)
                        rhos.append(0)
                        continue



                try:
                    MI.append(mutual_info_regression(ATAC_list[0][inboth, idxs[i]].todense(), RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2, n_neighbors=3, discrete_features=False))
                except ValueError:
                    #print("bad things happened")
                    MI.append(0)
                    rhos.append(0)
                    continue

                    #MI.append(mutual_info_regression(ATAC_list[0][inboth, idxs[i]].todense(), RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2, n_neighbors=1, discrete_features=False))

                unadjusted_rho = np.corrcoef(ATAC_list[0][inboth, idxs[i]].todense().transpose(), RNA_list[0][inboth, gene_idx].reshape(-1, 1).transpose())[0, 1]
                n = np.size(ATAC_list[0][inboth, idxs[i]].todense().transpose())
                adjusted_rh0 = unadjusted_rho * hyp2f1(1 / 2, 1 / 2, (n - 1) / 2, 1 - unadjusted_rho ** 2)
                rhos.append(unadjusted_rho)

                bar()

        MI = np.asarray(MI)
        rhos = np.asarray(rhos)


        # Check to see if the background distribution of Rhos is Gaussian mixture

        g_1 = GaussianMixture(n_components=1, random_state=0).fit(rhos.reshape(-1, 1))
        g_2 = GaussianMixture(n_components=2, random_state=0).fit(rhos.reshape(-1, 1))

        BIC_1 = g_1.bic(rhos.reshape(-1, 1))
        BIC_2 = g_2.bic(rhos.reshape(-1, 1))

        if BIC_1 <= BIC_2:
            print("Distribution of correlations of interest appear unimodal (Bayesian Info Criterion " + str(BIC_1) + " for gaussian vs. " + str(
                BIC_2) + " for mixture)")
        elif BIC_2 < BIC_1:
            print("Distribution of correlations of interest appear bimodal (Bayesian Info Criterion " + str(BIC_1) + " for gaussian vs. " + str(
                BIC_2) + " for mixture)")

        # Using Bonferroni Correction here
        n_peaks = np.size(MI)
        sig_cutoff = float(5)/float(n_peaks)
        sig_lvl = float(cutoff) / float(100)


        high_percentile_uncorrected = np.percentile(MI_bkg, 95)
        high_percentile_corr_uncorrected = np.percentile(rhos_bkg, 95)

        high_percentile = np.percentile(MI_bkg, 100 - sig_cutoff)
        high_percentile_corr = np.percentile(rhos_bkg, 100 - sig_cutoff)


        print('         ')
        print("Bonferroni Adjusted Percentile for Significance:")
        print(100 - sig_cutoff)
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

        print("Single Hypothesis MI for Significance:")
        print(high_percentile_uncorrected)
        print('         ')
        print("Single Hypothesis Correlation for Significance:")
        print(high_percentile_corr_uncorrected)
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

            try:

                if i in badguys:
                    pass
                    #print(str(peaks[i][0]) + '           ' + str(peaks[i][1]) + '       ' + str(MI[i][0]) + '          ' + 'too sparse' + '                ' + str(rhos[i]) + '         ' + 'too sparse ')
                else:
                    print(str(peaks[i][0]) + '           ' + str(peaks[i][1]) + '       ' + str(MI[i][0]) + '          ' +  txt_outs[i] + '                ' + str(rhos[i]) + '         ' + txt_corr_out[i])

            except:
                pass

        with open('corr_out_normalized.bedgraph', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, n_peaks):
                if i in badguys:
                    pass
                else:
                    writer.writerow([chrom, str(peaks[i][0]), str(peaks[i][1]), str(rhos[i])])

        with open('mi_out_normalized.bedgraph', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in range(0, n_peaks):
                if i in badguys:
                    pass
                else:
                    writer.writerow([chrom, str(peaks[i][0]), str(peaks[i][1]), str(MI[i][0])])

    else:
        ###############################################################
        ###############################################################
        ###############################################################
        ###############################################################
        ###############################################################
        # FULL MODE!!

        print("Running every gene in the .gtf file.")

        spinner = Halo(text='Prepare for liftoff...', spinner='dots',
                       color='white', placement='right')
        spinner.start()


        # ATAC list is [COUNTS, BARCODES, PEAKS, DEPTHS]
        ATAC_list = pickle.load(open(ATAC_pickled, "rb"))

        # RNA list is [COUNTS, BARCODES, GENE NAMES, GTF file]
        RNA_list = pickle.load(open(RNA_pickled, "rb"))

        # first we need to sample a background distribution of Mutual informations

        # pick random genes and random peaks

        n_genes = np.size(RNA_list[0], axis=1)
        n_peaks = np.size(ATAC_list[0], axis=1)

        gene_samples = 1
        peak_samples = 100

        MI_bkg = []
        rhos_bkg = []

        idxs_genes_bkg = np.random.randint(0, n_genes, gene_samples)

        spinner.stop()

        print("Calculating Mutual Information and Rho from Background")

        L = gene_samples * peak_samples

        with alive_bar(L) as bar:

            for j in range(0, gene_samples):

                gene_idx = idxs_genes_bkg[j]
                nonzero_rna = np.argwhere(RNA_list[0][:, gene_idx].todense() != 0)

                idxs_bkg = np.random.randint(0, n_genes, peak_samples)

                L = np.size(idxs_bkg)

                bad_peaks = 0

                for i in range(0, np.size(idxs_bkg)):

                    nonzero_atac = np.argwhere(ATAC_list[0][:, idxs_bkg[i]].todense() != 0)
                    inboth = np.intersect1d(nonzero_atac, nonzero_rna)

                    if np.size(inboth) is None:
                        bad_peaks += 1
                        bar()
                        continue

                    if np.size(inboth) <= cutoff:
                        bad_peaks += 1

                        bar()
                        continue

                    else:

                        # Need to filter out anything has stddev = 0
                        # dont think Im ever actually needing this
                        if np.std(np.asarray(ATAC_list[0][inboth, idxs_bkg[i]].todense().reshape(-1, 1), dtype=float)) == 0:
                            MI_bkg.append(0)
                            rhos_bkg.append(0)
                            bar()
                            continue
                        else:
                            try:
                                if np.std(np.asarray(RNA_list[0][inboth, gene_idx].reshape(-1, 1), dtype=float)) == 0:
                                    MI_bkg.append(0)
                                    rhos_bkg.append(0)
                                    bar()
                                    continue
                            except:
                                pass
                                # bar()

                        try:

                            MI_bkg.append(mutual_info_regression(ATAC_list[0][inboth, idxs_bkg[i]].todense(),
                                                                 RNA_list[0][inboth, gene_idx].reshape(-1, 1),
                                                                 random_state=2,
                                                                 n_neighbors=3, discrete_features=False))
                        except:
                            bar()
                            continue

                            # # Probably dont need this either
                            # # Also this is really sloppy since number of neighbors changes estimate
                            # print("A Mutual Information Calculation had n_samples < n_neighbors=3. Switched nn to 1.")
                            # MI_bkg.append(mutual_info_regression(ATAC_list[0][inboth, idxs_bkg[i]].todense(),
                            #                                      RNA_list[0][inboth, gene_idx].reshape(-1, 1),
                            #                                      random_state=2,
                            #                                      n_neighbors=1, discrete_features=False))

                        unadjusted_rho = np.corrcoef(ATAC_list[0][inboth, idxs_bkg[i]].todense().transpose(),
                                                     RNA_list[0][inboth, gene_idx].reshape(-1, 1).transpose())[0, 1]

                        n = np.size(ATAC_list[0][inboth, idxs_bkg[i]].todense().transpose())
                        adjusted_rho = unadjusted_rho * hyp2f1(1 / 2, 1 / 2, (n - 1) / 2, 1 - unadjusted_rho ** 2)
                        rhos_bkg.append(adjusted_rho)

                        bar()

        print("Calculating Mutual Information and Rho for Peaks around Genes")

        # Find all the gene locations then split across processes

        ###### SPECIFY SEARCH SIZE HERE
        search_size = 8000

        gene_dict = dict()

        locs_found = []
        chroms_found = []
        genes_found = []

        RNA_list[2] = np.char.lower(RNA_list[2])

        print("Organizing Genes")
        GTF_arr = RNA_list[4]
        for i in range(5, np.size(GTF_arr, axis=0)):
            if GTF_arr[i, 2] == 'gene':

                if i in range(20000, 23000):

                    # print(i)


                    print( GTF_arr[i, 8].split(' gene_name "')[1].split('"')[0] + str('                                            '), end="\r")
                    # print(np.isin(RNA_list[2], GTF_arr[i, 9].split(' gene_id "')[0][1:].split('"')[0]))

                    #if np.isin(RNA_list[2], GTF_arr[i, 9].split(' gene_id "')[0][1:].split('"')[0]) == True:
                    gene_idx = np.argwhere(RNA_list[2] == GTF_arr[i, 8].split(' gene_name "')[1].split('"')[0].lower())
                    if np.size(gene_idx) == 0:
                        continue


                    loc = int(GTF_arr[i, 3])

                    chromosome = GTF_arr[i, 0]

                    if chromosome not in gene_dict:
                        gene_dict[chromosome] = [[], [], [], []]
                    gene_dict[chromosome][0].append(int(loc))
                    gene_dict[chromosome][1].append(GTF_arr[i, 8].split(' gene_name "')[1].split('"')[0])

                    # While we're wasting time for looping this lets grab the indices of the peaks near
                    # these genes

                    idxs = np.intersect1d(np.argwhere(ATAC_list[2][:, 0] == chromosome), np.argwhere((np.asarray(ATAC_list[2][:, 1], dtype=int) > loc - search_size) & (np.asarray(ATAC_list[2][:, 1], dtype=int) <= loc + search_size)))

                    gene_dict[chromosome][2].append(idxs)
                    gene_dict[chromosome][3].append(gene_idx)

        spinner = Halo(text='Setting up to begin processes', spinner='dots',
                       color='white', placement='right')
        spinner.start()

        atac_chunks = dict()
        for chromosome in gene_dict.keys():
            chr_idxs = np.argwhere(ATAC_list[2][:, 0] == chromosome).flatten()
            sparse_chunk = scipy.sparse.lil_matrix(np.shape(ATAC_list[0]))
            sparse_chunk[:, chr_idxs] = ATAC_list[0][:, chr_idxs]
            atac_chunks[chromosome] = sparse_chunk

        spinner.stop()

        if __name__ != "__main__":

            with Manager() as manager:
                master_dict = manager.dict()
                processes = []
                for chromosome in gene_dict.keys():
                    print("started process for chrom " + str(chromosome))

                    p = Process(target=run_chrom, args=(chrom, gene_dict[chromosome], atac_chunks[chromosome], RNA_list[0], inboth, master_dict))

                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

                pickle.dump(master_dict, open( "big_outfile.p", "wb" ))




def run_chrom(chrom, list_of_list, atac_chunk, rna_counts, inboth, masterdict):

    mis = []
    rhos = []

    masterdict[chrom] = []

    for k in range(0, len(list_of_list[0])):

        print('                        ')
        print(k)

        gene_idx = list_of_list[3][k][0]
        idxs = list_of_list[2][k]

        nonzero_rna = np.argwhere(rna_counts[:, gene_idx].todense() != 0)
        nonzero_atac = np.argwhere(atac_chunk[:, idxs].todense() != 0)

        print(np.size(nonzero_rna))
        print(np.size(nonzero_atac))


        inboth = np.intersect1d(nonzero_atac, nonzero_rna)

        cutoff = 3
        if np.size(inboth) < 3:
            print(np.size(inboth))
            print("Too few for a gene")
            mis.append('-')
            rhos.append('-')
            continue

        try:
            mi = mutual_info_regression(atac_chunk[inboth, idxs].todense(),
                                   rna_counts[inboth, gene_idx].reshape(-1, 1), random_state=2,
                                   n_neighbors=3, discrete_features=False)
        except:
            print("went here")
            mi = 0


        mis.append(mi)




        rho = np.corrcoef(np.hstack(atac_chunk[inboth, idxs].todense().transpose(),
                                    rna_counts[inboth, gene_idx].reshape(-1, 1).transpose()))[-1, :]

        rhos.append(rho)

        # n = np.size(atac_chunk[inboth, idxs].todense().transpose())
        # adjusted_rho = rho * hyp2f1(1 / 2, 1 / 2, (n - 1) / 2, 1 - rho ** 2)
        # rhos.append(adjusted_rho)
        print("GOOD GENE!")

    masterdict[chrom].append([mis, rhos])
    return



def entrodecomp3(ATAC_pickled, RNA_pickled, gene=None, gene_2=None, chrom=None, start=None, stop=None, mode='expression', num_threads=1, window_size=30, cutoff = 10):

    # haven't implemented a background distribution for this yet


    window_size_kb = int(window_size)

    # ATAC list is [COUNTS, BARCODES, PEAKS, DEPTHS]
    ATAC_list = pickle.load(open(ATAC_pickled, "rb"))

    # RNA list is [COUNTS, BARCODES, GENE NAMES]
    RNA_list = pickle.load(open(RNA_pickled, "rb"))

    # Find the index of the gene we specified
    gene_idx = np.argwhere(RNA_list[2] == gene)[0]
    gene_2_idx = np.argwhere(RNA_list[2] == gene_2)[0]


    # find the peak that is within the specified range
    idxs = []
    peaks = []
    # Could vectorize this but fast anyway
    for i in range(0, np.size(ATAC_list[2], axis=0)):
        if ATAC_list[2][i, 0] == chrom:
            if int(ATAC_list[2][i, 1]) in range(start, stop):
                idxs.append(i)
                peaks.append([ATAC_list[2][i, 1], ATAC_list[2][i, 2]])
    idxs = np.asarray(idxs, dtype=int)

    if len(peaks) > 1:
        print("There is more than one peak in the specified region")
        raise ValueError
    if len(peaks) == 0:
        print("There are no peaks in the specified region")
        raise ValueError


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

    # print('         ')
    # print("Number of Background Peaks to Consider")
    # print(np.size(idxs_bkg))
    # print('         ')

    # Entropy decomposition for the chosen

    target_gene = RNA_list[0][:, gene_idx].todense()
    peak_data = (ATAC_list[0][:, idxs[0]]).todense()
    gene_2_data = RNA_list[0][:, gene_2_idx].todense()

    good_idxs = np.intersect1d(np.intersect1d(np.argwhere(target_gene != 0), np.argwhere(peak_data != 0)), np.argwhere(gene_2_data != 0))

    h_x, h_y, h_z, i_xy, i_xz, i_yz, i_xyGz, i_xzGy, i_yzGx, i_xyz = entropy_decomp_3(target_gene[good_idxs], peak_data[good_idxs], gene_2_data[good_idxs], n_neighbors=1)

    print('          ')
    print('2-Simplex Representation of Entropy Decomposition')
    print('  Note: # of samples is ' + str(np.size(good_idxs)))
    print('          ')

    print('3-Way Mutual Information')
    print(i_xyz)
    print('          ')
    print('Total Entropy')
    print(h_x + h_y + h_z)

    print('          ')
    print('          ')


    print('                 ' + gene + '                ')
    print('          ')
    print('                 ' + str(h_x) + '                ')
    print('                  / \                      ')
    print('                 /   \                     ')
    print('     ' + str(i_xyGz) + '       /     \     ' + str(i_xzGy) + '          ')
    print('               /       \                   ')
    print('              /         \                  ')
    print('             /___________\                 ')
    print('        '+str(h_y) +'     '  + str(i_yzGx) + '      ' + str(h_z) + '          ')
    print( '     peak                      ' + str(gene_2))

    print('          ')
    print('          ')

#
#
# gene_names = RNA_list[2]
#
# for i in range(0, n_genes):
#     gene_idx = gene_names[i]
#
#     # Now we go find the location of this gene in the GTF file
#     idxs = np.argwhere(RNA_list[4][:, 8] == "gene_id " + gene_names[i] + ";")
#     idx_start_codon = np.argwhere()
#
#
#
#
#     spinner = Halo(text='Calculating Mutual Information and Rho for Peaks of Interest', spinner='dots',
#                    color='white', placement='right')
#     spinner.start()
#
#     warnings.filterwarnings(action='ignore', category=DataConversionWarning)
#
#     # nonzero gene counts cells
#     nonzero_rna = np.argwhere(RNA_list[0][:, gene_idx].todense() != 0)
#
#     MI = []
#     rhos = []
#     badguys = []
#     for i in range(0, np.size(idxs)):
#
#         # Find indices of cells that have nonzero counts in both
#         nonzero_atac = np.argwhere(ATAC_list[0][:, idxs[i]].todense() != 0)
#         inboth = np.intersect1d(nonzero_atac, nonzero_rna)
#
#         if np.size(inboth) <= cutoff:
#             # print("Too sparse a peak to consider")
#             badguys.append(i)
#
#         # Need to filter out anything has stddev = 0
#         if np.std(np.asarray(ATAC_list[0][inboth, idxs[i]].todense().reshape(-1, 1), dtype=float)) == 0:
#             MI.append(0)
#             rhos.append(0)
#             continue
#         else:
#             if np.std(np.asarray(RNA_list[0][inboth, gene_idx].reshape(-1, 1), dtype=float)) == 0:
#                 MI.append(0)
#                 rhos.append(0)
#                 continue
#         try:
#             MI.append(mutual_info_regression(ATAC_list[0][inboth, idxs[i]].todense(),
#                                              RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2,
#                                              n_neighbors=3, discrete_features=False))
#         except ValueError:
#             MI.append(mutual_info_regression(ATAC_list[0][inboth, idxs[i]].todense(),
#                                              RNA_list[0][inboth, gene_idx].reshape(-1, 1), random_state=2,
#                                              n_neighbors=1, discrete_features=False))
#
#         unadjusted_rho = np.corrcoef(ATAC_list[0][inboth, idxs[i]].todense().transpose(),
#                                      RNA_list[0][inboth, gene_idx].reshape(-1, 1).transpose())[0, 1]
#         n = np.size(ATAC_list[0][inboth, idxs[i]].todense().transpose())
#         adjusted_rho = unadjusted_rho * hyp2f1(1 / 2, 1 / 2, (n - 1) / 2, 1 - unadjusted_rho ** 2)
#         rhos.append(adjusted_rho)
#
#     MI = np.asarray(MI)
#     rhos = np.asarray(rhos)
#
#     spinner.stop()
