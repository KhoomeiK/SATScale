# NOTE: ensure you increase --n_process to # CPUs in all the generate_*.py scripts

# sr data
python src/generate_sr_data.py data_nsnet/SATSolving/sr/train 3000000 --min_n 10 --max_n 40
python src/generate_sr_data.py data_nsnet/SATSolving/sr/valid 10000 --min_n 10 --max_n 40
python src/generate_sr_data.py data_nsnet/SATSolving/sr/test 10000 --min_n 10 --max_n 40

python src/generate_labels.py marginal data_nsnet/SATSolving/sr/train > /dev/null
python src/generate_labels.py marginal data_nsnet/SATSolving/sr/valid > /dev/null
python src/generate_labels.py marginal data_nsnet/SATSolving/sr/test > /dev/null

python src/generate_labels.py assignment data_nsnet/SATSolving/sr/train > /dev/null
python src/generate_labels.py assignment data_nsnet/SATSolving/sr/valid > /dev/null
python src/generate_labels.py assignment data_nsnet/SATSolving/sr/test > /dev/null

# 3-sat data
python src/generate_3-sat_data.py data_nsnet/SATSolving/3-sat/train 2000000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py data_nsnet/SATSolving/3-sat/valid 10000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py data_nsnet/SATSolving/3-sat/test 10000 --min_n 10 --max_n 40

python src/generate_labels.py marginal data_nsnet/SATSolving/3-sat/train > /dev/null
python src/generate_labels.py marginal data_nsnet/SATSolving/3-sat/valid > /dev/null
python src/generate_labels.py marginal data_nsnet/SATSolving/3-sat/test > /dev/null

python src/generate_labels.py assignment data_nsnet/SATSolving/3-sat/train > /dev/null
python src/generate_labels.py assignment data_nsnet/SATSolving/3-sat/valid > /dev/null
python src/generate_labels.py assignment data_nsnet/SATSolving/3-sat/test > /dev/null

# ca data
python src/generate_ca_data.py data_nsnet/SATSolving/ca/train 2000000 --min_n 10 --max_n 40
python src/generate_ca_data.py data_nsnet/SATSolving/ca/valid 10000 --min_n 10 --max_n 40
python src/generate_ca_data.py data_nsnet/SATSolving/ca/test 10000 --min_n 10 --max_n 40

python src/generate_labels.py marginal data_nsnet/SATSolving/ca/train > /dev/null
python src/generate_labels.py marginal data_nsnet/SATSolving/ca/valid > /dev/null
python src/generate_labels.py marginal data_nsnet/SATSolving/ca/test > /dev/null

python src/generate_labels.py assignment data_nsnet/SATSolving/ca/train > /dev/null
python src/generate_labels.py assignment data_nsnet/SATSolving/ca/valid > /dev/null
python src/generate_labels.py assignment data_nsnet/SATSolving/ca/test > /dev/null

python src/upload_data.py data_nsnet/SATSolving