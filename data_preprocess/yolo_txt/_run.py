import runpy

runpy.run_path('_check_empty.py') # set to move version

# correct and add the labels back

runpy.run_path('rename_files.py') # initial rename, remove roboflow

# move ot active files

runpy.run_path('train_valid_split.py') # split train 100 to train/valid


# after re-labeling is complete, run the below
runpy.run_path('_default_class_correction.py')
runpy.run_path('_remove_roboflow.py')
