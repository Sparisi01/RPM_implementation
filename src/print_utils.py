# print_utils.py

WIDTH = 70

def print_section(title, width=WIDTH, char='-', new_line=True):
    if new_line:
        print(f"\n{title.center(width, char)}\n")
    else:
        print(f"{title.center(width, char)}")


def print_line(width=WIDTH, char='-', new_line=False):
    print_section(title="", width=width, char=char, new_line=new_line)


def print_kv(key, value, width=WIDTH, value_width=25):
    print(f"{key:<{width - value_width - 1}} {value:>{value_width}}")


WIDTH = 70


def print_table_header(headers: list[str]):

    sep = " | "
    total_sep_width = len(sep) * (len(headers) - 1)

    col_width = (WIDTH - total_sep_width) // len(headers)
    
    print_line(char='=')

    print(
        " | ".join(
            f"{h:^{col_width}}"
            for h in headers
        )
    )

    print_line()


def print_table_row(values: list):

    sep = " | "
    total_sep_width = len(sep) * (len(values) - 1)

    col_width = (WIDTH - total_sep_width) // len(values)

    print(
        " | ".join(
            f"{str(v):>{col_width}}"
            for v in values
        )
    )