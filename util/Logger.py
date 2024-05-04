import shutil
import textwrap

def print_centered(content, fill='='):
    # Get the terminal width
    terminal_width = shutil.get_terminal_size().columns
    
    # Ensure there's space for at least one fill character on each side plus a space for the content
    max_content_width = terminal_width - 4
    
    # Split the content into lines that fit within the max_content_width
    wrapped_content = textwrap.wrap(content, max_content_width)
    
    for line in wrapped_content:
        # Calculate the length of the line plus two spaces for padding
        line_length = len(line) + 2
        
        # Calculate the number of fill characters on both sides
        padding_length = (terminal_width - line_length) // 2
        
        # Create the full line to print
        full_line = fill * padding_length + ' ' + line + ' ' + fill * padding_length
        
        # Adjust if terminal_width isn't exactly divisible by 2
        if len(full_line) < terminal_width:
            full_line += fill * (terminal_width - len(full_line))
        
        print(full_line)


def fill_row(fill):
    # Get the terminal width
    terminal_width = shutil.get_terminal_size().columns
    
    # Create and print a line filled with the specified character
    print(fill * terminal_width)