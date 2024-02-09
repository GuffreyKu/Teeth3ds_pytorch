import os

def folder_check():
    folder_paths = ['saved_model',
                    'eval_fig',
                    'part_teeth',
                    'data']

    for path in folder_paths:
        if not os.path.exists(path):
            os.mkdir(path)