python src/train_model.py model-counting mc_nsnet_satlib ~/scratch/NSNet/ModelCounting/SATLIB/train/ --valid_dir ~/scratch/NSNet/ModelCounting/SATLIB/valid/ --epochs 1000 --scheduler StepLR --lr_step_size 200
python src/test_model.py model-counting ~/scratch/NSNet/ModelCounting/SATLIB/test/ --checkpoint runs/mc_nsnet_satlib/checkpoints/model_best.pt 