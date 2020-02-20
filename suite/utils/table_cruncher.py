import re
import pandas as pd
from common.utils.tables import TABLES


LATEX_PRE = '\\begin{table}\n\t\\centering\n\t\\resizebox{\\textwidth}{!}{'
LATEX_POST = '}\label{{label}}\caption{{caption}) \%}\end{table}'

LATEX_TABLE_PRE = '\\begin{table}\n\t\\centering\n\t\\resizebox{\\textwidth}{!}{\n'
LATEX_TABLE_POST = '}\\end{table}'
LATEX_PARBOX_PARBOX = '\t\\parbox{.5\\textwidth}{\n\t\t\\resizebox{0.5\\textwidth}{!}{%\n'
LATEX_POST_PARBOX = '}%\n'

OUTPUT = '../../Dokumentation/src/common/tables/tables.tex'


def p2f(x):
    return float(x.strip('%')) / 100


def f2p(x):
    return '{:.2%}'.format(x)


def f2pb(x):
    return '{:.2%}'.format(x)


def f2i(x):
    return int(float(x))


def f2is(x):
    return str(f2i(x))


if __name__ == '__main__':

    pd.set_option('display.precision', 2)
    pd.set_option('display.float_format', '{:.2f}'.format)

    with open(OUTPUT, 'w', encoding='utf-8') as f:

        for t, table in enumerate(TABLES):

            if t % 2 == 0:
                f.write(LATEX_TABLE_PRE)
            else:
                f.write('\\qquad')

            f.write(LATEX_PARBOX_PARBOX)

            label = table.get('label', 'NO LABEL')
            caption = table.get('caption', 'NO CAPTION')
            calculate_group_average = table.get('calculate_group_average', False)

            table_names = []
            for p, table_details in enumerate(table.get('tables', [])):
                table_name, path = table_details
                table_names.append(table_name)

            target_header = pd.MultiIndex.from_frame(pd.DataFrame([
                ['Sharp Force', 'n'], ('Sharp Force', 'Precision'), ['Sharp Force', 'Recall'], ['Sharp Force', 'F1'], ['Sharp Force', 'AP'],
                ['Blunt Force', 'n'], ['Blunt Force', 'Precision'], ['Blunt Force', 'Recall'], ['Blunt Force', 'F1'], ['Blunt Force', 'AP'],
                ['', 'mAP']
            ]), names=['', ''])
            target_index = pd.MultiIndex.from_product([
                (0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
                table_names + (['Avg',] if calculate_group_average else [])
            ], names=['IoU', 'Network'], )

            target_table = pd.DataFrame(
                index=target_index,
                columns=target_header,
                dtype='float32'
            )

            group_size = len(table.get('tables', [])) + (1 if calculate_group_average else 0)

            for p, table_details in enumerate(table.get('tables', [])):

                table_name, path = table_details

                eval_table = pd.read_csv(
                    path,
                    index_col='Class',
                    header=0,
                    converters={
                        'n': f2i,
                        'Precision': p2f,
                        'Recall': p2f,
                        'mAP': p2f,
                    }
                )

                for iou in range(6):
                    target_table.iloc[p + iou * group_size, :5] = eval_table.iloc[iou, 1:].values
                    target_table.iloc[p + iou * group_size, 5:10] = eval_table.iloc[iou + 6, 1:].values

            if calculate_group_average:
                for iou in range(6):
                    target_table.iloc[iou * group_size + group_size - 1, :] = target_table.iloc[iou * group_size: iou * group_size + group_size - 1, :].mean(axis=0)

            target_table[('', 'mAP')] = target_table.iloc[:, target_table.columns.get_level_values(1) == 'AP'].mean(axis=1)
            latex = '{table}'.format(
                table=target_table.to_latex(
                    formatters=[f2is, f2p, f2p, None, f2p, f2is, f2p, f2p, None, f2p, f2pb]
                ).replace('end{tabular}', 'end{tabular}%'))

            last_name = 'Avg' if calculate_group_average else table_names[-1]
            latex = re.sub(
                r'({}.*)(\d+\.\d+\\%)(\s\\\\)'.format(last_name),
                r'\1{}\3 \\hline'.format(r'\\textbf{\2}' if calculate_group_average else r'\2'),
                latex
            )
            latex = re.sub(r'(IoU\s&\sNetwork.*\\\\)', r'\1 \\hline', latex)
            latex = re.sub(r'[lr]{4,}', r'l|l|lllll|lllll|l', latex)

            f.write(latex)
            f.write(LATEX_POST_PARBOX)
            f.write('\\caption{{{}}}\n\\label{{tab:{}}}\n'.format(caption, label))
            f.write('}')

            if t > 0 and ((t + 1) % 2 == 0 or (t + 1) >= len(TABLES)):
                f.write(LATEX_TABLE_POST)

            if t > 0 and (t + 1) % 12 == 0:
                f.write('\clearpage')
