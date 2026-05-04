# print_utils.py>


def print_section(title, width=70, char='-'):
    print(f"\n{title.center(width, char)}\n")
    
def print_kv(key, value, width=70):
    print(f"{key:<60} {value:>9}")