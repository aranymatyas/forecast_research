#!/usr/bin/env python3
"""
Process a raw markdown table file for inclusion in the paper.

Steps applied:
1. Remove columns: error_metric, method
2. Remove column: count (unless filename contains 'pos' or 'neg')
3. Rename columns: granularity -> D/M, forecast_type -> type
4. Truncate all numeric values to 3 decimal places
5. Remove leading zeros from exponents (e+07 -> e+7)
6. Replace hyphens with non-breaking hyphens (U+2011) in data cells
7. Remove whitespace from type column values
8. Strip whitespace from all cells
9. Adjust separator dash counts to match max content width per column
10. Wrap with \\footnotesize, Table: title \\label{name}, \\normalsize

Usage:
    python process_table.py <file.md> [--title "Table title"]
    python process_table.py path/to/stats.md
"""
import sys
import os
import re
import argparse

NON_BREAKING_HYPHEN = '\u2011'
NUM_RE = re.compile(r'-?\d+\.\d+(?:e[+-]?\d+)?')
EXP_RE = re.compile(r'e([+-]?)0+(\d+)')
COLUMNS_TO_REMOVE_ALWAYS = {'error_metric', 'method'}
COLUMN_RENAMES = {'granularity': 'D/M', 'forecast_type': 'type'}
VALUE_RENAMES = {
    'reconciled_forecasts': 'Reco',
    'base_forecasts': 'Base',
    'reconciled_forecasts - base_forecasts': 'Reco‑Base',
    'reconciled_forecasts - reconciled_forecasts': 'Reco‑Reco',
    'Days': 'D',
    'Months': 'M',
}


def truncate_num(match):
    s = match.group(0)
    if 'e' in s:
        mantissa, exp = s.split('e')
    else:
        mantissa, exp = s, None
    if '.' in mantissa:
        int_part, dec_part = mantissa.split('.')
        mantissa = f'{int_part}.{dec_part[:3]}'
    return f'{mantissa}e{exp}' if exp else mantissa


def fix_exp(match):
    sign = match.group(1) or '+'
    digits = match.group(2).lstrip('0') or '0'
    return f'e{sign}{digits}'


def process_table(filepath, title=None):
    basename = os.path.basename(filepath)
    label = os.path.basename(os.path.dirname(os.path.abspath(filepath))) + '/' + basename.replace('.md', '')
    has_pos_neg = 'pos' in basename or 'neg' in basename

    with open(filepath) as f:
        content = f.read()

    # Strip any existing wrapper (in case re-processing)
    content = content.replace('\\footnotesize', '').replace('\\normalsize', '')
    lines = [l for l in content.strip().splitlines() if not l.startswith('Table:')]
    lines = [l for l in lines if l.strip()]

    # Find table lines
    table_lines = [l for l in lines if l.startswith('|') or l.strip().startswith('|')]
    if len(table_lines) < 2:
        print(f"ERROR: No valid table found in {filepath}")
        return

    header_line = table_lines[0]
    sep_line = table_lines[1]
    data_lines = table_lines[2:]

    # Parse header
    header_cells = [c.strip() for c in header_line.split('|')]

    # Rename columns
    header_cells = [COLUMN_RENAMES.get(c, c) for c in header_cells]

    # Determine columns to remove
    cols_to_remove = set()
    for i, c in enumerate(header_cells):
        if c in COLUMNS_TO_REMOVE_ALWAYS:
            cols_to_remove.add(i)
        if c == 'count' and not has_pos_neg:
            cols_to_remove.add(i)

    # Remove columns from header
    header_cells = [c for i, c in enumerate(header_cells) if i not in cols_to_remove]

    # Parse separator to preserve alignment
    sep_cells = [c.strip() for c in sep_line.split('|')]
    sep_cells = [c for i, c in enumerate(sep_cells) if i not in cols_to_remove]

    # Process data lines
    processed_data = []
    for line in data_lines:
        cells = [c.strip() for c in line.split('|')]
        cells = [c for i, c in enumerate(cells) if i not in cols_to_remove]
        processed_data.append(cells)

    # Find type column index (after removal)
    type_idx = None
    for i, c in enumerate(header_cells):
        if c == 'type':
            type_idx = i
            break

    # Process cell values
    for row in processed_data:
        for j in range(len(row)):
            cell = row[j]
            # Apply value renames
            cell = VALUE_RENAMES.get(cell, cell)
            # Truncate decimals
            cell = NUM_RE.sub(truncate_num, cell)
            # Fix exponents
            cell = EXP_RE.sub(fix_exp, cell)
            # Replace hyphens with non-breaking hyphens
            cell = cell.replace('-', NON_BREAKING_HYPHEN)
            # Remove whitespace from type column
            if j == type_idx:
                cell = cell.replace(' ', '')
            row[j] = cell

    # Also fix hyphens in header (except empty cells)
    header_cells = [c.replace('-', NON_BREAKING_HYPHEN) if c and c not in ('', 'D/M') else c for c in header_cells]

    # Build separator with proportional dashes
    max_widths = [len(c) for c in header_cells]
    for row in processed_data:
        for j, c in enumerate(row):
            if j < len(max_widths):
                max_widths[j] = max(max_widths[j], len(c))

    new_sep_cells = []
    for j, cell in enumerate(sep_cells):
        if not cell or not re.match(r'^:?-+:?$', cell):
            new_sep_cells.append(cell)
            continue
        left = cell.startswith(':')
        right = cell.endswith(':')
        width = max(3, max_widths[j] if j < len(max_widths) else 3)
        dashes = '-' * (width - int(left) - int(right))
        new_sep_cells.append((':' if left else '') + dashes + (':' if right else ''))

    # Assemble table
    header_out = '|'.join(header_cells)
    sep_out = '|'.join(new_sep_cells)
    data_out = '\n'.join('|'.join(row) for row in processed_data)

    # Title
    if not title:
        title = 'title'

    table = f'\\footnotesize\n\n{header_out}\n{sep_out}\n{data_out}\n\nTable: {title} \\label{{{label}}}\n\n\\normalsize\n'

    with open(filepath, 'w') as f:
        f.write(table)

    print(f'Processed: {filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process raw markdown table for paper inclusion')
    parser.add_argument('file', help='Path to .md table file')
    parser.add_argument('--title', default=None, help='Table caption (default: "title")')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    process_table(args.file, args.title)
