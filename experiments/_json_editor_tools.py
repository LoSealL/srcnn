import json
import argparse
from pathlib import Path


def AddCommonKeyValue(entry, value):
    pass


def DeleteCommonKey(entry):
    pass


def ModCommonKeyValue(entry, value):
    pass


parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, help='file1 file2... [Can be \'all\']')
parser.add_argument('--add', type=str, required=False, default=None,
                    help='Key1:Key2:Key3:...:KeyN=Value')
parser.add_argument('--rm', type=str, required=False, default=None,
                    help='Key')
parser.add_argument('--mod', type=str, required=False, default=None,
                    help='Key1:Key2:Key3:...:KeyN=Value')

args, arg_lists = parser.parse_known_args()
# collect files
files = [args.files] + arg_lists
all_files = Path('.').glob('*.json')
valid_file = all([Path(i).exists() for i in files])
if files[0] == 'all':
    files = all_files
elif not valid_file:
    raise RuntimeError('detect invalid filename!')
# parse ops
if args.add or args.mod:
    op = args.add or args.mod
    entries, value = op.split('=')
    entry = entries.split(':')
    if value.isdigit():
        value = int(value)
    elif value.lower() == 'false':
        value = False
    elif value.lower() == 'true':
        value = True
if args.rm:
    entry = args.rm

for file in files:
    fd = open(file, 'r')
    param = json.load(fd)
    assert isinstance(param, dict)
    if args.add or args.mod:
        assert entry, value
        tgt = param
        for e in entry[:-1]:
            try:
                tgt = param[e]
            except KeyError:
                param[e] = {}
                tgt = param[e]
        tgt[entry[-1]] = value
    if args.rm:
        try:
            param.pop(entry)
        except KeyError as ex:
            pass
    fd.close()
    fd = open(file, 'w')
    json.dump(param, fd, indent=2, sort_keys=True)
