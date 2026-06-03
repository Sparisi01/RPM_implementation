# print_utils.py

WIDTH = 70

def print_section(title: str, width: int = WIDTH, char: str='-', new_line: bool=True):
    if new_line:
        print(f"\n{title.center(width, char)}\n")
    else:
        print(f"{title.center(width, char)}")


def print_line(width: int=WIDTH, char: str='-', new_line: bool = False):
    print_section(title="", width=width, char=char, new_line = new_line)


def print_kv(key, value, width: int=WIDTH, value_width: int=25):
    print(f"{key:<{width - value_width - 1}} {value:>{value_width}}")

def print_table_header(headers: list[str], width: int = WIDTH, sep: str = " | "):
    
    total_sep_width = len(sep) * (len(headers) - 1)
    col_width = (width - total_sep_width) // len(headers)
    print(
        " | ".join(
            f"{h:^{col_width}}"
            for h in headers
        )
    )

def print_table_row(values: list, width: int = WIDTH, sep: str = " | "):

    total_sep_width = len(sep) * (len(values) - 1)

    col_width = (width - total_sep_width) // len(values)

    print(
        " | ".join(
            f"{str(v):>{col_width}}"
            for v in values
        )
    )