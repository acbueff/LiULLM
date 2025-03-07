#SRC=/p/data1/trustllmd/ebert1/baseline-data-llm-foundry/
SRC=/p/data1/trustllmd/filatov1/data/baseline-llm-foundry/
DST=ai030477@transfer1.bsc.es:/gpfs/scratch/ehpc09/baseline-data-llm-foundry
#rsync -av --append-verify --include="/*/train/" --include="/*/val/" --exclude="*.mds" --exclude="/*/*/"  $SRC $DST 
rsync -av --append-verify --include="/*-fast/train/" --include="/*-fast/val/" --exclude="/*/*/"  $SRC $DST 
