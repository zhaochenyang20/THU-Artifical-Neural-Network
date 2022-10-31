import os
import sys
import traceback
import re
import argparse
import difflib
import pprint

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", type=str, default=os.path.join("..", "codes"))
parser.add_argument("--origin_dir", type=str, default="_codes")
parser.add_argument("--output_file", type=str, default="summary.txt")
args = parser.parse_args()

differ = difflib.Differ()

def mapping(dir1, dir2):
    file_maps = {}
    fset1 = set(os.listdir(dir1))
    fset2 = set(os.listdir(dir2))
    missing = fset1.difference(fset2)
    additional = fset2.difference(fset1)
    for fname in fset1.intersection(fset2):
        fname1 = os.path.join(dir1, fname)
        fname2 = os.path.join(dir2, fname)
        if os.path.isdir(fname1):
            _file_maps, _missing, _additional = mapping(fname1, fname2)
            file_maps.update(_file_maps)
            missing.update(_missing)
            additional.update(_additional)
        else:
            file_maps[fname1] = fname2
    return file_maps, missing, additional

def analyze_filled_code(text_pairs, stdout):
    references = set()
    print("########################", file=stdout)
    print("# Filled Code", file=stdout)
    print("########################", file=stdout)
    for pair in text_pairs:
        fname1, text1 = pair[0]['fname'], pair[0]['text']
        fname2, text2 = pair[1]['fname'], pair[1]['text']
        res = list(differ.compare(text1.splitlines(True), text2.splitlines(True)))
        idx = 0
        block_id = 0
        block = []
        in_block = False
        out_block = True
        while idx < len(res):
            line = res[idx]
            if out_block and re.search(r'^(-| )\s*# TODO START', line):
                block_id += 1
                block = []
                in_block = True
                out_block = False
            elif in_block and re.search(r'^(\+| )\s*# TODO END', line):
                in_block = False
                out_block = True
                print("# {}:{}".format(fname2, block_id), file=stdout)
                for line in block:
                    print("{}".format(line.rstrip()), file=stdout)
                print("", file=stdout)
            elif in_block and re.search(r'^\+\s*# Reference:', line):
                ref = re.search(r'^\+\s*# Reference:\s*(?P<ref>\S.+\S)\s*$', line)
                if ref and "e.g." not in ref.group("ref"):
                    references.add(ref.group("ref"))
                block.append(line[2:])
            # elif in_block and re.search(r'^-\s*# TODO END', line):
            #    raise RuntimeError('Do not remove or change the line "# TODO END" in your file')
            elif in_block and re.search(r'^\+\s*# TODO START', line) is None and line[0] == '+':
                block.append(line[2:])
            idx += 1
    print(file=stdout)
    print("########################", file=stdout)
    print("# References", file=stdout)
    print("########################", file=stdout)
    for ref in references:
        print("# {}".format(ref), file=stdout)
    print(file=stdout)

def extract_modifications(text_pairs, stdout):
    print("########################", file=stdout)
    print("# Other Modifications", file=stdout)
    print("########################", file=stdout)
    for pair in text_pairs:
        fname_printed = False
        fname1, text1 = pair[0]['fname'], pair[0]['text']
        fname2, text2 = pair[1]['fname'], pair[1]['text']
        lno1, lno2 = 0, 0
        res = list(differ.compare(text1.splitlines(True), text2.splitlines(True)))
        idx = 0
        in_block = False
        end_block = False
        out_block = True
        while idx < len(res):
            line = res[idx]
            lno1 += int(line[0] in [' ', '-'])
            lno2 += int(line[0] in [' ', '+'])
            if out_block and re.search(r'^(-|\+| )\s*# TODO START', line):
                in_block = True
                out_block = False
            elif in_block and re.search(r'^(-|\+| )\s*# TODO END', line):
                end_block = True
                in_block = False
            elif end_block and re.search(r'^(-|\+| )\s*# TODO END', line) is None:
                end_block = False
                out_block = True
            if out_block and (line[0] in ['-', '+'] or (line[0] == '?' and re.search(r'^(-|\+| )\s*# TODO END', res[idx - 1])) is None):
                if not fname_printed:
                    print("# {} -> {}".format(fname1, fname2), file=stdout)
                    fname_printed = True
                lno = lno2
                if line[0] in ['-'] or (line[0] in ['?'] and res[idx - 1][0] in ['-']):
                    lno = lno1
                print("# {}".format(lno), line.rstrip(), file=stdout)
            idx += 1
    print(file=stdout)

def main():
    file_maps, missing_files, additional_files = mapping(args.origin_dir, args.target_dir)
    with open(args.output_file, "w", encoding='utf-8') as file:

        if missing_files:
            print("########################", file=file)
            file.write("# Missing Files\n")
            print("########################", file=file)
            for fname in missing_files:
                file.write("# {}\n".format(fname))
            file.write("\n")

        if additional_files:
            print("########################", file=file)
            file.write("# Additional Files\n")
            print("########################", file=file)
            for fname in additional_files:
                file.write("# {}\n".format(fname))
            file.write("\n")

        core_text_pairs = []
        text_pairs = []
        for fname1, fname2 in file_maps.items():
            try:
                replace_leading_tab = lambda s: re.sub(r"^\t+", lambda m: "    "*len(m.group()), s, flags=re.M)
                text1 = replace_leading_tab(open(fname1, "r", encoding='utf-8').read())
                text2 = replace_leading_tab(open(fname2, "r", encoding='utf-8').read())
            except Exception as err:
                print(traceback.format_exc())
                sys.exit(1)
            text_pair = (
                {'fname': fname1, 'text': text1},
                {'fname': fname2, 'text': text2}
            )
            if re.search(r'# TODO START[\S\s]*# TODO END', text1):
                core_text_pairs.append(text_pair)
            text_pairs.append(text_pair)
        analyze_filled_code(core_text_pairs, file)
        extract_modifications(text_pairs, file)

if __name__ == '__main__':
    main()