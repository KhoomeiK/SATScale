# sr data
python src/generate_sr_data.py data_nsnet/SATSolving/sr/train 30000 --min_n 10 --max_n 40
python src/generate_sr_data.py data_nsnet/SATSolving/sr/valid 10000 --min_n 10 --max_n 40
python src/generate_sr_data.py data_nsnet/SATSolving/sr/test 10000 --min_n 10 --max_n 40
python src/generate_sr_data.py data_nsnet/SATSolving/sr/test_hard 10000 --min_n 40 --max_n 200

python src/generate_labels.py marginal data_nsnet/SATSolving/sr/train
python src/generate_labels.py marginal data_nsnet/SATSolving/sr/valid

python src/generate_labels.py assignment data_nsnet/SATSolving/sr/train
python src/generate_labels.py assignment data_nsnet/SATSolving/sr/valid

# 3-sat data
python src/generate_3-sat_data.py data_nsnet/SATSolving/3-sat/train 30000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py data_nsnet/SATSolving/3-sat/valid 10000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py data_nsnet/SATSolving/3-sat/test 10000 --min_n 10 --max_n 40
python src/generate_3-sat_data.py data_nsnet/SATSolving/3-sat/test_hard 10000 --min_n 40 --max_n 200

python src/generate_labels.py marginal data_nsnet/SATSolving/3-sat/train
python src/generate_labels.py marginal data_nsnet/SATSolving/3-sat/valid

python src/generate_labels.py assignment data_nsnet/SATSolving/3-sat/train
python src/generate_labels.py assignment data_nsnet/SATSolving/3-sat/valid

# ca data
python src/generate_ca_data.py data_nsnet/SATSolving/ca/train 30000 --min_n 10 --max_n 40
python src/generate_ca_data.py data_nsnet/SATSolving/ca/valid 10000 --min_n 10 --max_n 40
python src/generate_ca_data.py data_nsnet/SATSolving/ca/test 10000 --min_n 10 --max_n 40
python src/generate_ca_data.py data_nsnet/SATSolving/ca/test_hard 10000 --min_n 40 --max_n 200

python src/generate_labels.py marginal data_nsnet/SATSolving/ca/train
python src/generate_labels.py marginal data_nsnet/SATSolving/ca/valid

python src/generate_labels.py assignment data_nsnet/SATSolving/ca/train
python src/generate_labels.py assignment data_nsnet/SATSolving/ca/valid

python src/upload_data.py data_nsnet/SATSolving