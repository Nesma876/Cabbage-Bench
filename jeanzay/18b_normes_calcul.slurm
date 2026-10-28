#!/bin/bash
#SBATCH --job-name=normes
#SBATCH --account=rme@h100
#SBATCH --gres=gpu:1
#SBATCH -C h100
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%a_%j.out
#SBATCH --error=logs/%x_%a_%j.err
set -euo pipefail
BASE=/lustre/fswork/projects/rech/rme/ufb85tk/pfe_tlm
source "$BASE/env-pfe/bin/activate"
cd "$BASE/cabbage_bench"
mkdir -p logs
NORMES=(clsi eucast)
export CABBAGE_NORME=${NORMES[$SLURM_ARRAY_TASK_ID]}
export CABBAGE_MATRICE=amr
export CABBAGE_PARQUET=$BASE/parquet/2025-11
SORTIE=$BASE/bench_norme_$CABBAGE_NORME
test -s "$CABBAGE_PARQUET/phenotype.parquet" || { echo "etape A non faite"; exit 1; }
echo "=== norme=$CABBAGE_NORME debut $(date)"
python run_all.py --sortie "$SORTIE" --fils 8 \
    --especes "Klebsiella pneumoniae" "Acinetobacter baumannii" \
              "Pseudomonas aeruginosa" "Staphylococcus aureus" \
              "Escherichia coli" "Salmonella enterica"
python baselines_forts.py --bench "$SORTIE"
python geographie.py      --bench "$SORTIE"
echo "=== fin $(date)"
ls -l "$SORTIE"/*.csv
