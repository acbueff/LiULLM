import json
from dataclasses import dataclass, asdict
from pathlib import Path

LANGS = json.loads(open('langs.json').read())
ROOT = Path("/p/scratch/trustllm-eu/WP2")

ISO639 = {}
for name, codes in LANGS.items():
    assert 'iso639_3' in codes
    canonical_code = codes['iso639_3']
    for c in set(codes.values()):
        assert c is not None
        assert c not in ISO639
        ISO639[c] = canonical_code

def to_iso639_3(l):
    return ISO639[l.lower()]

@dataclass(frozen=True)
class source:
    lang: str
    path: str
    source: str
    text_getter: str = '.text'

def get_fhg():
    rets = []
    # OSCAR
    root = ROOT / 'FHG' / 'web' / 'OSCAR_CLEAN_DEDUP'
    for p in filter(lambda x: x.is_dir(), root.iterdir()):
        lang = to_iso639_3(p.name)
        if lang not in ['eng', 'deu']:
            rets += [source(lang, str(file), 'oscar_clean_dedup') for file in (p / 'oscar').glob('*/*.jsonl')]
        else:
            rets += [source(lang, str(file), 'oscar_clean_dedup') for file in p.glob('*/*.jsonl')]

    # CURATED
    root = ROOT / 'FHG' / 'curated' / 'v3_by_lang'
    for p in filter(lambda x: x.is_dir(), root.iterdir()):
        lang = to_iso639_3(p.name)
        for k in filter(lambda x: x.is_dir(), p.iterdir()):
            kind = k.name
            rets += [source(lang, str(file), kind) for file in k.glob('*.jsonl')]

    return rets

def get_ais():
    rets = []
    # HPLT
    root = ROOT / 'AI-Sweden' / 'trustpipe' / 'data' / 'github-com-trustllmeu-trove-git-main-crawl-hplt-ingest_0029e96233478cd06281c8d1ef7212dc' / 'cleaned'
    for p in filter(lambda x: x.is_dir(), root.iterdir()):
        lang = to_iso639_3(p.name)
        rets += [source(lang, str(file), 'hplt_v.1.2') for file in p.glob('*json*')]
    return rets
        
    # CELLAR
    root = ROOT / 'AI-Sweden' / 'trustpipe' / 'data' / 'github-com-trustllmeu-trove-git-main-sources-cellar-process_17dc43e7219dc0a3dc01a43d699e9fa6'
    for f in filter(lambda x: x.is_dir(), root.iterdir()):
        for l in filter(lambda x: x.is_dir(), f.iterdir()):
            lang = to_iso639_3(l.name)
            fmt = f.name
            rets += [source(lang, str(file), 'cellar', text_getter='.data') for file in l.glob('*json.gz')]

    # PATENT
    root = ROOT / 'AI-Sweden' / 'trustpipe' / 'data' / 'github-com-trustllmeu-trove-git-main-sources-prv-process_8ad8fe1a5bcd7c81c15d3a915c8161a8'
    rets += [
            source(to_iso639_3('sv'), str(root / 'sv.json'), 'prv'),
            source(to_iso639_3('en'), str(root / 'en.json'), 'prv'),
            ]
    
    return rets


def get_alexandra():
    root = ROOT / 'alexandrainst' / 'datasets'
    lang = to_iso639_3('da')

    rets = []

    for d in Path(root).iterdir():
        if d.name == 'scandi-wiki':
            continue # Disregard this, already covered

        for file in (d / 'documents').glob('*.gz'):
            rets.append(source(lang, str(file), d.name))
    return rets


def get_uoi():
    lang = 'fao'
    rets = [
        source(lang, str(ROOT / 'UoI' / 'FC3.jsonl.gz'), 'FC3'),
        source(lang, str(ROOT / 'UoI' / 'Faroese_BLARK_small.jsonl.gz'), 'Faroese_BLARK_small')
    ]
    return rets

def get_norwai():
    root = ROOT / 'NorwAI' 
    lang = 'nor'
    rets = []
    IGNORE = ['nowac', 'texts_from_norwegian_wikipedia'] # lots of unicode errors 
    for d in Path(root).iterdir():
        if d.name == 'public_directories':
            for sd in d.iterdir():
                for file in sd.glob('*.json'):
                    rets.append(source(lang, str(file), f'public_directory_{sd.name}'))
        elif d.name not in IGNORE:
            for file in (d / 'output').glob('*.json'):
                rets.append(source(lang, str(file), d.name))
    return rets

def get_mideind():
    # Directory per source
    igc_dir = ROOT / "mideind" / "IGC"
    ic3v2_dir = ROOT / "mideind" / "IC3v2"
    blog_is_dir = ROOT / "mideind" / "blog_is"
    hugi_dir = ROOT / "mideind" / "hugi"
    rafbokavefurinn_dir = ROOT / "mideind" / "rafbokavefurinn"
    mim_dir = ROOT / "mideind" / "mim"
    fundargerdir_borgarrads_dir = ROOT / "mideind" / "fundargerdir_borgarrads"
    studentabladid_dir = ROOT / "mideind" / "studentabladid"
    eea_dir = ROOT / "mideind" / "eea"
    skemman_dir = ROOT / "mideind" / "skemman-pdf-nov-2022"

    # Getters for the text
    igc_getter = r'.document | map(join(" ")) | join("\n")'
    ic3v2_getter = r".document"
    blog_is_getter = r".content"
    hugi_getter = r".content"
    rafbokavefurinn_getter = r'.document | map(join(" ")) | join("\n")'
    mim_getter = r'.document | map(join(" ")) | join("\n")'
    fundargerdir_borgarrads_getter = r".document"
    studentabladid_getter = r'.document | map(join(" ")) | join("\n")'
    eea_getter = r'.document | map(join(" ")) | join("\n")'
    skemman_getter = r'.document | map(join(" ")) | join("\n")'

    lang = to_iso639_3("is")

    rets = []
    rets += [
        source(lang, str(file), "IGC", text_getter=igc_getter)
        for file in igc_dir.rglob("*.json*")
    ]
    rets += [
        source(lang, str(file), "IC3v2", text_getter=ic3v2_getter)
        for file in ic3v2_dir.rglob("*.json*")
    ]
    rets += [
        source(
            lang,
            str(blog_is_dir / "items_cleaned.jsonl"),
            "blog_is",
            text_getter=blog_is_getter,
        )
    ]
    rets += [
        source(
            lang,
            str(hugi_dir / "hugi-korkar.jsonl"),
            "hugi",
            text_getter=hugi_getter,
        )
    ]  # TODO: Concat with the reply chain
    rets += [
        source(
            lang,
            str(rafbokavefurinn_dir / "rafbokavefurinn.all.detokenized.jsonl"),
            "rafbokavefurinn",
            text_getter=rafbokavefurinn_getter,
        )
    ]
    rets += [
        source(lang, str(file), "mim", text_getter=mim_getter)
        for file in mim_dir.rglob("*.json*")
    ]
    rets += [
        source(
            lang,
            str(fundargerdir_borgarrads_dir / "borgarrad_texts.jsonl"),
            "fundargerdir_borgarrads",
            text_getter=fundargerdir_borgarrads_getter,
        )
    ]
    rets += [
        source(
            lang,
            str(studentabladid_dir / "json_dump/studentabladid_is_train.jsonl"),
            "studentabladid",
            text_getter=studentabladid_getter,
        )
    ]
    rets += [
        source(
            lang, str(eea_dir / "json_dump/eea_is.jsonl"), "eea", text_getter=eea_getter
        )
    ]
    rets += [
        source(
            lang,
            str(skemman_dir / "jsonl/skemman_filtered_only_is_no_references.jsonl"),
            "skemman",
            text_getter=skemman_getter,
        )
    ]

    return rets


sources = []
sources += get_fhg()
sources += get_ais()
sources += get_uoi()
sources += get_mideind()
sources += get_alexandra()
sources += get_norwai()

for s in set(sources):
    print(json.dumps(asdict(s)))
