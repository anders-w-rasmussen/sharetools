#!/usr/bin/env python


import argparse
import sys
from sharetools.preprocess import run_preprocess
from sharetools.MI import mutual_information


class sharetools(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Information Theory Tools for Exploring Regulatory Dynamics from SHARE-Seq Data',
            usage=

            '''sharetools <command> [<args>]
    
                Commands:
    
                    preprocess           Preprocess SHARE-seq data from count matrices
                
                    mutual_info          Calculate mutual information values
                                         
                    entropy_decomp3      Entropy decomposition for set of two genes and a peak
                
                
                Future things to consider implementing:
                
                    entropy decompositions of much larger sets and inform graph construction
                    
                    do dimensionality reductiona and then entropy calculations
            
                    ''')

        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def preprocess(self):
        formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=55)
        parser = argparse.ArgumentParser(formatter_class=formatter,
                                         description='Preprocess SHARE-seq data from count matrices',
                                         usage='''sharetools preprocess <RNA_count_file> <ATAC_count_file> <ATAC_barcode_file> <ATAC_peak_file> <GTF_file> [-h] [options]
            ''')

        parser.add_argument("RNA_count_file", help='RNA_counts', action='store', type=str)
        parser.add_argument("ATAC_count_file", help='ATAC_counts', action='store', type=str)
        parser.add_argument("ATAC_barcode_file", help='ATAC_barcodes',
                            action='store', type=str)
        parser.add_argument('ATAC_peak_file', help='ATAC_peaks (bed)', action='store', type=str)
        parser.add_argument('GTF_file', help='Reference GTF File', action='store', type=str)
        parser.add_argument('--max_threads', help='maximum number of cores this command can use', required=False, default=1, action='store', type=int,
                            dest='max_threads')
        parser.add_argument('--velocity_file', help='file containing RNA-velocity information', required=False,
                            action='store', type=str, dest='velocity_file')
        parser.add_argument('--outname', help='prefix for outfiles', required=False,
                            action='store', type=str, dest='outname')

        args = parser.parse_args(sys.argv[2:])

        run_preprocess(args.RNA_count_file, args.ATAC_count_file, args.ATAC_barcode_file,
                args.ATAC_peak_file, args.GTF_file, args.max_threads, args.velocity_file, args.outname)


    def mutual_info(self):
        formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=55)
        parser = argparse.ArgumentParser(formatter_class=formatter,
                                         description='Calculate Mutual Information between Peaks and a Gene',
                                         usage='''sharetools mutual_info <ATAC_file> <RNA_file> [-h] [options]
            ''')

        parser.add_argument("ATAC_file", help='Preprocessed ATAC file (pickled)', action='store', type=str)
        parser.add_argument("RNA_file", help='Preprocessed RNA file (pickled)', action='store', type=str)
        parser.add_argument('--gene', help='gene of interest', required=False,
                            default=None, action='store', type=str,
                            dest='gene')
        parser.add_argument('--chrom', help='chromosome', required=False,
                            default=None, action='store', type=str,
                            dest='chrom')
        parser.add_argument('--start', help='start base on chromosome', required=False,
                            default=None, action='store', type=int,
                            dest='start')
        parser.add_argument('--stop', help='stop base on chromosome', required=False,
                            default=None, action='store', type=int,
                            dest='stop')
        parser.add_argument('--mode', help='mode', required=False,
                            default=None, action='store', type=str,
                            dest='mode')
        parser.add_argument('--num_threads', help='threads', required=False,
                            default=None, action='store', type=int,
                            dest='num_threads')
        parser.add_argument('--bkg_dist', help='distance from region within which bkg calculated', required=False,
                            default=None, action='store', type=int,
                            dest='bkg_dist')
        parser.add_argument('--cutoff', help='distance from region within which bkg calculated', required=False,
                            default=None, action='store', type=int,
                            dest='cutoff')


        args = parser.parse_args(sys.argv[2:])

        mutual_information(args.ATAC_file, args.RNA_file, args.gene,
                args.chrom, args.start, args.stop, args.mode, args.num_threads, args.bkg_dist, args.cutoff)


if __name__ == '__main__':
    sharetools()