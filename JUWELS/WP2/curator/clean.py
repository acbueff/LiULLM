import pathlib

root = pathlib.Path('/p/data1/trustllmd/WP2/data/signals')

files = list(root.rglob('TMP*.parquet'))

if files:

    for file in files:
        print(file)

    ans = input('delete? [yes|no]')

    if ans == 'yes':
        print('deleting')
        for file in files:
            file.unlink()

else:
    print('nothing to delete')

