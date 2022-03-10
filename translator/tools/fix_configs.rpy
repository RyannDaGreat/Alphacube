def get_all_config_files():
    return get_all_files(file_extension_filter='yaml',recursive=True)
def add_line_to_all_config_files(line:str):
    for path in get_all_config_files():
        append_line_to_file(line,path)
        print('Added line to',path)