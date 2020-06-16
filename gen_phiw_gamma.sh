
# bash script to generate the results with different gamma values

fn_inp_ref='inp_ref.py'
out_path='data_SPHS_gamma_DTPs'

mkdir ${out_path}
mkdir ${out_path}/analysis


# testing the different gamma value
# gamma=1: hard spheres
# gamma<1: solvent-permeable hard spheres
for gamma in 1.000 0.979 0.888 0.763
do
    gname=${gamma//./} # removing dot from gamma
    fn_inp_gref='inp_g'${gname}'.py'
    awk '{ if ($1 == "gamma") $3="'${gamma}'"; print $0 }' ${fn_inp_ref} > data_SPHS_gamma_DTPs/${fn_inp_gref}

    # automatically testing with different DLP    
    for DTP in $(seq -w 5000 500 6000) # -w option keeps the digit with leading zero
    do
	# DTP=5000
	echo $gamma
	# echo $gname
	echo $DTP
	fn_inp='inp_g'${gname}'_DTP'${DTP}'.py'	
	fn_out='phiw_g'${gname}'_DTP'${DTP}'.dat'
	awk '{ if ($1 == "ref_DTP") $3="'${DTP}'"; print $0 }' ${out_path}/${fn_inp_gref} > ${out_path}/${fn_inp}
	python ../scripts/cal_phiw_from_input.py ${out_path}/${fn_inp} ${out_path}/${fn_out}
	python plot_analysis.py ${out_path}/${fn_out} ${out_path}/analysis/${fn_out}.pdf
    done
done


