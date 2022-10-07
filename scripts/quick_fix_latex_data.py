import re
import argparse

def fix(file):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    filtered_lines = []

    def fst_or_null(l):
        return None if l is None else l[1]

    filtered_lines.append(lines[0])
    for i in range(1, len(lines) - 1):
        s1 = fst_or_null(re.search(r"""\(\d+, (\d+)\)""", lines[i-1]))
        s2 = fst_or_null(re.search(r"""\(\d+, (\d+)\)""", lines[i]))
        s3 = fst_or_null(re.search(r"""\(\d+, (\d+)\)""", lines[i+1]))
        if s1 != s2 or s3 is None or s2 != s3:
            filtered_lines.append(lines[i])
    filtered_lines.append(lines[-1])

    with open(file, 'w') as f:
        f.write(''.join(filtered_lines))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('file')

    args = p.parse_args()
    fix(args.file)