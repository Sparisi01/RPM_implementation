# print_utils.py>

def print_section(title, width=70, char='-', new_line = True):
    if new_line:
        print(f"\n{title.center(width, char)}\n")
    else:
        print(f"{title.center(width, char)}")
    
    
def print_line(width=70, char='-', new_line = False):
    print_section(title="", width= width, char=char, new_line = new_line)
    
def print_kv(key, value, width=70, value_width = 15):
    print(f"{key:<{width-value_width-1}} {value:>{value_width}}")