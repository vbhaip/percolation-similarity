# Percolation Similarity

This repo contains my work for the OMMS (Oxford Masters in Math) dissertation, supervised by Professor Renaud Lambiotte. It evaluates a new type of similarity metric on networks measuring information flow. The full dissertation can be read at [paper.pdf](paper.pdf).

## Getting Started

`python3 simulate.py --help`: See different arguments you can pass in.

### Examples

`python3 simulate.py --metric all -g karate --one_plot`
Runs all the metrics on Zachary's Karate Club Network. Output all the plots on one graphic. 

`python3 simulate.py --metric jaccard,resistance -g clique --clique_size 5 -d graphs`
Run Jaccard Index and Resistance Distance metrics on Clique of size 5. Output to a directory called 'output/graphs'.

`python3 simulate.py --metric all -g core_periphery --cp_sizes "[20,20]" --cp_probs "[[0.9, 0.5], [0.5, 0.1]]" --seed 7`
Runs all metrics on a stochastic block model graph with two communities of sizes 20 and 20, with the specified probability block matrix. The seed is a random seed for generating the realization of the graph from the stochastic block model. 


### Outputs

Each command outputs a directory with its name as the day and time the program was run (unless the `--dir` flag is passed in). In addition to outputting the heat maps, there are a few more files:
- `command.txt`: Command run to generate the data in the folder. Helpful to see what generated the output and possibly re-run program.
- `metric_data.txt`: Time of execution in seconds for computing each metric's similarity values for all the pairs of nodes for the specified graph.
- `rank_differences.txt`: Pairwise comparison of metrics specified for where the largest differences in the ranked heat maps are. For example, suppose metric 1 ranks some node pair having a similarity of 1, and metric 2 ranks that same node pair with similarity 200. If this is one of the top ten biggest differences in the rankings, the nodes would be listed along with the value for each of the similarity rankings. Useful for seeing how different metrics disagree.

### Commands Used to Generate Plots in Paper

- `python3 simulate.py --metric all -g clique --clique_size 6 --one_plot`
- `python3 simulate.py --metric all -g connected_cliques --clique_size 6 --num_cliques 6 --one_plot`
- `python3 simulate.py --metric all -g clique_missing_edge --clique_size 6 --one_plot`
- `python3 simulate.py --metric all -g clique_with_outsider --clique_size 6 --one_plot`
- `python3 simulate.py --metric all -g core_periphery --cp_sizes "[20,20]" --cp_probs "[[0.9, 0.5], [0.5, 0.1]]" --one_plot`
- `python3 simulate.py --metric all -g core_periphery --cp_sizes "[10,10,10]" --cp_probs "[[0.95, 0.5, 0.3], [0.5, 0.1, 0.1], [0.3, 0.1, 0.05]]" --one_plot`
- `python3 simulate.py --metric all -g karate --one_plot`
- `python3 simulate.py --metric all -g florentine --one_plot`


