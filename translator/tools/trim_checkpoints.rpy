#This is meant to be used in the model outputs folder

def get_all_checkpoint_folders(outputs_folder='.'):
    for output_folder in get_subfolders(outputs_folder):
        checkpoint_folder=path_join(output_folder,'checkpoints')
        if folder_exists(checkpoint_folder):
            yield checkpoint_folder
        
def delete_old_checkpoints(folder='.'):
    files=get_all_files(folder,file_extension_filter='pt',sort_by='date')
    old_files=files[:-4]
    for file in old_files:
        delete_file(file)
        print('Deleted:',file)
