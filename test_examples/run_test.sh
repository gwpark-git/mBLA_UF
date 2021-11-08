fn_inp='inp_v3.py'
fn_out='test_ref.dat'
python ../scripts/cal_phiw_from_input.py $fn_inp $fn_out
python ../analysis/plot_analysis.py inp_ref.py test_ref.dat test_ref.pdf
