#!/bin/bash
set -e

dataset="/project/bioinformatics/DLLab/STUDIES/ABIDE1/Derivatives/Freesurfer_rerun"
output="$dataset"

module load freesurfer/stable_v6.0.0

echo "listing subjects..."
subjects=($dataset/*/stats/aseg.stats)
subjects=("${subjects[@]%/stats/aseg.stats}")
subjects=("${subjects[@]##*/}")
echo "done"

SUBJECTS_DIR=$dataset asegstats2table --subjects ${subjects[@]} --skip --delimiter="comma" -m volume -t $output/aseg.csv
SUBJECTS_DIR=$dataset aparcstats2table --subjects ${subjects[@]} --skip --delimiter="comma" --hemi lh -m area -t $output/lh_area.csv
SUBJECTS_DIR=$dataset aparcstats2table --subjects ${subjects[@]} --skip --delimiter="comma" --hemi lh -m thickness -t $output/lh_thickness.csv
SUBJECTS_DIR=$dataset aparcstats2table --subjects ${subjects[@]} --skip --delimiter="comma" --hemi rh -m area -t $output/rh_area.csv
SUBJECTS_DIR=$dataset aparcstats2table --subjects ${subjects[@]} --skip --delimiter="comma" --hemi rh -m thickness -t $output/rh_thickness.csv