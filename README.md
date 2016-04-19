# Locational Referential Expressions

Parsing referential expressions for the Baxter research robot. A class project for Brown University CSCI2971K, by Roshan Rao, Eli Sharf, and Edward Williams. 

### Visualizations

All visualizations can be run via `data_vis.py`. All default visualizations can be run from the command line.

##### Visualizing Point Clouds

This visualization shows the objects on the board, twelve point clouds for the different commands (color coded by command), and the naive estimation of the "true point." To see this visualization, run the following command

	./data_vis.py <scene_num>

To save this visualization as a pdf to a file, run

	./data_vis.py <scene_num> save <filename>

##### Visualizing Distributions for a Given Algorithm

This visualization shows twelve subplots, one for each command. Each subplot shows the objects, the data from the given command, and the output distribution for the given command and algorithm. To see this visualization, run the folloiwng command

	./data_vis.py <scene_num> <algorithm>

Where algorithm is one of `'cheating', 'naive', 'objects', 'objects_walls', 'refpt'`. To save this visualization as a pdf to a file, run

	./data_vis.py <scene_num> <algorithm> save <filename>

Note that it is also possible to view the distribution for a single command and algorithm via `visualize_distribution`. See how `visualize_all_distributions` in `data_vis.py` calls `visualize_distribution` to determine how to do this.

##### Algorithm Evaluation

There are two algorithm evaluation measures. The first one evaluates the negative log probability of the product of datapoints from the distribution that a given algorithm generates. That is, if we let `f_alg(pt | command, world)` be the output pdf of an algorithm given a command and a world, this evaluation shows the following quantity

	-sum(log(f_alg(pt | command, world) for pt in dataset[world][command]))

To see a bar chart displaying this quantity for each algorithm and command for a given scene, run

	./algorithm_evaluation.py <scene_num>

The second algorithm evaluation measure shows the distance between the estimated mean of the data given by each algorithm and the empirical mean of the data. To see this evaulation run

	./algorithm_evaluation.py <scene_num> means

To save either of these to a file as a pdf, add `save <filename>` to the end.


##### Creating All Default Visualizations for a Scene

The shell script `create_vis.sh` automatically creates all of the default distributions and saves them in the `visualizations` folder. Just run 

	create_vis.sh <scene_num>

This will show you all the distributions and save them.