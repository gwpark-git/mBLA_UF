# Generating phiw with different DTP conditions
# DLP = 130
# DTP = 130, 250, 500, ..., 4750, 5000
#
# this script is related with generating the conditions with different DTP
# then execute the python scripts for the different DTP
# The format of the reference input file is of importance since 'awk' commend will be used in this script
# the default field separator is ' ' (space), which means
# "a = 100" treated as	$0 for whole sentence,
#			$1 for the first field which is a	
#			$2 for the second field which is =				
#			$3 for the third field which is 100				
# therefore, it is not recommendable to modify any white-space based on input file.
#

fn_inp_ref='inp_ref.py'
fn_base='MODI_DLP130_DTP'

# the first test condition

DTP=130
fn_inp='inp_'${fn_base}${DTP}'.py'
fn_out='phiw_'${fn_base}${DTP}'.dat'
awk '{ if ($1 == "ref_DTP") $3="'${DTP}'"; print $0 }' ${fn_inp_ref} > data/${fn_inp}
python scripts/cal_phiw_from_input.py data/${fn_inp} data/${fn_out}
python plot_analysis.py data/${fn_out} data/analysis/${fn_out}.pdf

# automatically testing with different DTP

for DTP in $(seq 500 250 5000)
do
    fn_inp='inp_'${fn_base}${DTP}'.py'
    fn_out='phiw_'${fn_base}${DTP}'.dat'
    awk '{ if ($1 == "ref_DTP") $3="'${DTP}'"; print $0 }' ${fn_inp_ref} > data/${fn_inp}
    python scripts/cal_phiw_from_input.py data/${fn_inp} data/${fn_out}
    python plot_analysis.py data/${fn_out} data/analysis/${fn_out}.pdf    
done


