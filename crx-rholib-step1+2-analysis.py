"""
Notebook to analyze step1 or step2 of crx RhoLib cloning
step1 has just CRE-BC
step2 has CRE-BC-rBC

makes plots of BC distribution
and CRE-BC agreements
generates an output file
if there are multiple samples - can run parse-output.ipynb to combine all the output files
"""

import collections
import gzip
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import pathlib
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

def bc_cre_agree_disagree(df, agree_or_disagree, read_type, outl):
    """
    Get % of valid BCs and valid CREs that agree or disagree
    Input
        df: dataframe with these columns: lib_id_from_cre, lib_id_from_bc
        agree_or_disagree: str either 'agree' or 'disagree'
        read_type: str describing the reads, 'merged' for raw reads, 'consensus' for rBC consensus
        outl: list used to generate output file
    Output
        modifies outl in place and prints result
    """
    
    # to do
    pass


def make_output(t, v, outl):
    """
    Prints out a description and a value
    Also adds tuple of (description, value) to a list for later printing to output file
    Input
        t: str, text description
        v: value
    Output
        modifies outl in place and prints
    """
    print(f"{t}: {v}")
    outl.append((t,v))

def reverse_complement(s):
    """
    Reverse complement of DNA
    Input
        s: str, DNA sequence
    """
    rc_dict = {"A":"T","C":"G","G":"C","T":"A","N":"N"}
    return "".join([rc_dict[i] for i in s[::-1]])

sample_name = "s1v1"
fq_file = f"/Users/cohenlab/Desktop/2022-05-26-step1-v1-spikein/merged.fq.gz"
has_rbc = False
before_fix = True

output_file = pathlib.Path(f"output.{sample_name}.txt")
outl = []

rc = True

# the promoter is normally 137 bp
# but there are sequences with added spacer up to 20 bp (and then add 1 to prevent BbsI sites from touching)
# the following pattern matches just the exact sizes
exact_size_pattern = re.compile("CCACCTGGTCGAGATATCT(\w{137,157})TCAGTGGTCTTC\w{0,21}GAAGACACGACG(\w{10})TCGACGGTACTCGATACTC")

# here is a pattern that allows for some size deviations in the promoter
loose_size_pattern_without_rbc = re.compile("CCACCTGGTCGAGATATCT(\w{1,170})TCAGTGGTCTTC\w{0,21}GAAGACACGACG(\w{10})TCGACGGTACTCGATACTC")

# special case before I fixed the 1 bp deletion in plasmid
loose_size_pattern_without_rbc_before_fix = re.compile("CCACCTGGG{0,1}TCGAGATATCT(\w{1,170})TCAGTGGTCTTC\w{0,21}GAAGACACGACG(\w{10})TCGACGGTACTCGATACTC")

# pattern that has extended region for the random barcode (also has size deviations in promoter)
loose_size_pattern_with_rbc = re.compile("\w{0,4}CCACCTGGTCGAGATATCT(\w{1,170})TCAGTGGTCTTC\w{0,21}GAAGACACGACG(\w{10})TCGACGGTACTCGATACTCCCG(\w{8})CA(\w{8})AGGACGACTCTATCAGTCGG\w{0,4}")

# read in the designed library
agilent_file = "/Users/cohenlab/Box/crx-rho-deep-mutagenesis/2022-03-18-library-design/rho-mutagenesis-library-output/RhoDeepMutLib.csv"
agilent_df = pd.read_csv(agilent_file)
agilent_df["index_col"] = agilent_df.index # I use this later in the merge to get the "ID" of the library member

# replace the 10 N's with the barcode sequence
agilent_sequences = agilent_df.apply(lambda x: x.agilent_seq.replace("N"*10, x.barcode1), axis=1)

# double check the regex matches all library sequences
for seq in agilent_sequences:
    m = exact_size_pattern.search(seq)
    if not m:
        print(seq)
        break
else:
    print("Regex matched all library sequences")
    
# make a set of the barcodes
agilent_bc_set = set(agilent_df.barcode1.values)

# read in the merged reads as a df
rows = []
total_merged_reads = 0

if not has_rbc:
    if before_fix:
        pattern = loose_size_pattern_without_rbc_before_fix
    else:
        pattern = loose_size_pattern_without_rbc
else:
    pattern = loose_size_pattern_with_rbc
    
with gzip.open(fq_file, mode="rt") as f:
    for line in f:
        total_merged_reads += 1
        sequence = next(f).strip()
        if rc:
            sequence = reverse_complement(sequence)
        next(f)
        next(f)
        
        m = pattern.search(sequence)
        if m:
            read = sequence
            cre = m.group(1)
            bc = m.group(2)
            
            if not has_rbc:
                rows.append(dict(read=read, read_cre=cre, read_bc=bc,))
            else:
                rbc1 = m.group(3)
                rbc2 = m.group(4)
                rbc = rbc1 + "CA" + rbc2
                rows.append(dict(read=read, read_cre=cre, read_bc=bc, rbc=rbc)) 

df = pd.DataFrame(rows)
loose_regex_matches = df.shape[0]/total_merged_reads*100
make_output("total # of merged reads", total_merged_reads, outl)
make_output("# of merged reads that match the regex allowing variable-sized CREs", df.shape[0], outl)
make_output("% of merged reads that match the regex allowing variable-sized CREs", loose_regex_matches, outl)

# merge the dataframes based on the barcode
mdf = df.merge(agilent_df[["barcode1", "index_col"]], how="left", left_on="read_bc", right_on="barcode1").drop(columns="barcode1")
mdf = mdf.rename(columns={"index_col":"lib_id_from_bc"})

percent_true_bcs = mdf.lib_id_from_bc.count() / len(mdf) * 100
# note that .count() counts non-NA cells
make_output("% of barcodes that match a library barcode", percent_true_bcs, outl)

# see what % of the libary BCs were covered
percent_lib_bc_covered = len(mdf.lib_id_from_bc.dropna().unique()) / len(agilent_bc_set) * 100
make_output("% of library barcodes covered", percent_lib_bc_covered, outl)

# merge the dataframes based on the CRE
mdf = mdf.merge(agilent_df[["seq", "index_col"]], how="left", left_on="read_cre", right_on="seq").drop(columns="seq")
mdf = mdf.rename(columns={"index_col":"lib_id_from_cre"})

percent_true_cres = mdf.lib_id_from_cre.count() / len(mdf) * 100
make_output("% of cres that match a library cre", percent_true_cres, outl)

bc_cre_agree_disagree(mdf, "agree", "merged", outl)
bc_cre_agree_disagree(mdf, "disagree", "merged", outl)

# Look at barcode distribution

fig, ax = plt.subplots()
ax.hist(mdf.lib_id_from_bc.value_counts().values, bins=100);

# trying to follow this guide for KDE/bandwidth estimation
# but it is a bit out of date since some of the imports are different now
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
x = mdf.lib_id_from_bc.value_counts().values
x_d = np.linspace(0,x.max(),1000)
bandwidth = 1
kde = KernelDensity(bandwidth = bandwidth, kernel= "gaussian")
kde.fit(x[:, None])

logprob = kde.score_samples(x_d[:, None])
fig, ax = plt.subplots()
ax.fill_between(x_d, np.exp(logprob), alpha=0.5,)
#ax.plot(x_d, np.exp(logprob))
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
shift_text_left = xmax/50
shift_text_up = ymax/50

percentiles = (50,75,90,95,99,100)
line_height = ymax/10 # height of vlines

vlines = [np.percentile(x,i) for i in percentiles]
ax.vlines(vlines, line_height, "k")
for vline, percentile in zip(vlines, percentiles):
    ax.text(vline - shift_text_left, line_height + shift_text_up, percentile)
    
ax.set_ylim((ymin, ymax))
ax.set_title(f"Distribution of reads per barcode\n{sample_name}\nbandwidth: {bandwidth}")
ax.set_xlabel("# of reads for barcode")
fig.tight_layout()
plt.savefig(f"{sample_name}.bc-count-dist.png")

