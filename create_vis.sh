#!/bin/bash
if [[ $# -eq 1 ]]; then
	SCENE_NUM="$1"
	[[ -e ./visualizations/scene_$SCENE_NUM ]] || mkdir ./visualizations/scene_$SCENE_NUM
	./data_vis.py $SCENE_NUM 				save ./visualizations/scene_$SCENE_NUM/scene_$SCENE_NUM.pdf

	./data_vis.py $SCENE_NUM cheating 		save ./visualizations/scene_$SCENE_NUM/alg_cheating.pdf
	./data_vis.py $SCENE_NUM naive 			save ./visualizations/scene_$SCENE_NUM/alg_naive.pdf
	./data_vis.py $SCENE_NUM objects 		save ./visualizations/scene_$SCENE_NUM/alg_objects.pdf
	./data_vis.py $SCENE_NUM objects_walls 	save ./visualizations/scene_$SCENE_NUM/alg_objects_walls.pdf
	./data_vis.py $SCENE_NUM refpt 			save ./visualizations/scene_$SCENE_NUM/alg_refpt.pdf

	./algorithm_evaluation $SCENE_NUM 		save ./visualizations/scene_$SCENE_NUM/distribution_evaluation.pdf
	./algorithm_evaluation $SCENE_NUM means save ./visualizations/scene_$SCENE_NUM/mean_evaluation.pdf
else
	echo "Usage: create_vis.sh <SCENE_NUM>" >&2
	exit 1
fi
