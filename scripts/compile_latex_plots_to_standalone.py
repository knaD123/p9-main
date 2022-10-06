import os
import argparse
import glob
import quick_fix_latex_data


p = argparse.ArgumentParser()

p.add_argument('dir', help='Directory containing .tex plot files')

args = p.parse_args()

plot_conf = {
    'memory_failure': r'ylabel={Average Connectedness}, xlabel={Memory Usage per Router per Demand}, ymode=log, xtick={2,3,9,15,20,25},ytick = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}, yticklabels = {$1-10^{-3}$, $1-10^{-4}$, $1-10^{-5}$, $1-10^{-6}$, $1-10^{-7}$},y dir=reverse,',
    'fwd_gen_time': r'ylabel={Avg. Data Plane Gen. Time per Demand}, xlabel={Topology},ymode=log, xlabel={Topology},label style={font=\scriptsize},',
    'latency': r'ylabel={Median number of hops}, xlabel={Demand},label style={font=\scriptsize}, ymin=1, ymax=140, ytick={1, 10, 100, 140}, yticklabels={1, 10, 100, $\infty$}, ymode=log,',
}

os.chdir(args.dir)

for f in glob.glob('*.tex'):
    if 'padded' in f or not any(k in f for k in plot_conf.keys()):
        continue
    with open(f, 'r') as original:
        a = f'{f[:-4]}_padded.tex'
        with open(a, 'w') as new:
            g = next(c for k, c in plot_conf.items() if k in f)

            s = """\\documentclass[margin=10,varwidth]{standalone}\\usepackage[utf8]{inputenc}\\usepackage{amsmath} \\usepackage{amsfonts} \\usepackage{amssymb} \\usepackage{xcolor} \\usepackage{tikz} \\usepackage{pgfplots}
                    \\begin{document}\\definecolor{dgreen}{rgb}{0.0, 0.6, 0.0}

                    \\begin{tikzpicture}
                    \\begin{axis}[
                        legend style = {anchor=north,
                                at={(0.5,-0.2)}, font=\\small},
                    """ + g + """
                    legend columns=3, tick label style={font=\\scriptsize}, minor y tick style = {draw = none}, y label style = {yshift = -5pt}, legend style = {font=\\large, column sep=5mm}, height=9cm
                    ]
                    """ + original.read() + """
                    \\end{axis} \\end{tikzpicture} \\end{document}"""

            new.write(s)
            if 'latency' in a:
                quick_fix_latex_data.fix(a)

        os.system(f'pdflatex {os.path.basename(a)}')
        os.system('rm *.aux *.log')
