# istartsc2021
Code repository for the paper "Investigating Strategies Towards Adversarially-Robust Time Series Classification" (published in Pattern Recognition Letters)

# Required Packages
Please install the following packages (via conda or pip as desired) before running:
1. Tensorflow 2.x
2. tslearn
3. pyts
4. texttable
5. scikit-learn
6. matplotlib

# 1. Data Preparation
Before running any of the provided scripts, a data preparation step is required. This involves:
1. The UCR TS Archive **(2015)**, available from [here](https://www.cs.ucr.edu/~eamonn/time_series_data/)
2. The Pretrained ResNet models provided by Ismail Fawaz et al, available from [here](https://germain-forestier.info/src/ijcnn2019/pre-trained-resnet.zip)
3. The adversarial data provided by Ismail Fawaz (available on request from Ismail Fawaz et. al, see paper [here](https://arxiv.org/pdf/1903.07054.pdf))

Each of these must be extracted into the root of this source tree. Afterwards, the `consolidate_datasets.py` script must be run, with the path to the UCR TS archive data directory (1 above) and the adversarial data directory (3 above) specified as command line arguments. This will automatically create a `Data` directory in the root of this source tree containing the prepared data.

# 2. Code Structure
The code is structured around the experiments/plots presented in the paper. For example, `fig1.py` generates Figure 1 as seen in the paper. Note that all scripts must be run from the root of this source tree, to follow the default path configuration.

## Hypothesis 1
The scripts in the `Hypothesis 1` directory are concerned with replicating the results around Hypothesis 1 in the paper. By default, the scripts only display the associated graphs. If the data constituting the graphs is needed, the associated lines can be uncommented.

## Hypothesis 2
The scripts in the `Hypothesis 2` directory are concerned with replicating the results around Hypothesis 2. By default, the scripts save the relevant plots to the root of the source tree. If the raw data is needed, the associated lines can be uncommented.

## Hypothesis 3
The scripts in the `Hypothesis 3` directory are for the learning shapelets and ROCKET TS classifiers - they're not meant to be run directly since Hypothesis 3 has to do with all the datasets in the UCR TS Archive, rather than individual datasets (i.e as in Hypotheses 1 and 2). To carry out the full suite of experiments for Hypothesis 3, invoke the `run_tests.py` script in the root of this source tree. This will automatically invoke the relevant files from the Hypothesis 3 directory and creates a `Results` directory to store the individual results. The results per dataset are named using a simple scheme: "{CLASSIFIERCODE}\_{dataset}\_results.txt", where CLASSIFIERCODE is either "LTS" or "ROCKET" (can be edited from the relevant classifier files in the Hypothesis 3 directory). Each file contains the classification accuracy for each of the n runs of the relevant classifier (n can be configured from the run_tests.py script (i.e the n\_runs variable).

Once the `run_tests.py` script completes, the results directory (configured in the run_tests.py file, in the _pathToResults_ variable) will contain result files named according to the convention described above. 

In order to view the results for Hypothesis 3 (i.e Figures 5-9 in the paper), the following postprocessing steps must be followed:

1. Run the `result_analyzer.py` script, with the path to the result directory (by default, it is "." i.e the current directory) as its command line argument. It will ask for the desired classifier code. Use the appropriate classifier code for the results/graphs of interest e.g ROCKET for the ROCKET classifier, LTS for the learned shapelets classifier. It will also ask which dataset is desired (by default, use * for all datasets)

2. The script will produce 2 result files in the root of the source tree: one pretty-printed tabular result (named {CLASSIFIERCODE}-results-pretty.txt) and a CSV-formatted result file (named {CLASSIFIERCODE}-results-csv.txt)

3. The actual plots can be obtained by invoking the `result_plots.py` script, with two arguments: the first being the path to the CSV result file (as generated from Step 2 above) and which particular plot is desired (0 for all, 1-4 for specific plots documented in the code)

By default, the individual plots are displayed then saved to disk when the `result_plots.py` script is run. To save the actual plot data, uncomment the relevant portions of the code.

The data for the final plot (i.e for sensitivity analyses) can be obtained by running the `result_stats.py` script. The script will prompt for the classifier code, in a bid to load the `{CLASSIFIERCODE}-results-csv.txt` file (which should be already generated from Step 2 above, in the current directory i.e the root of this source tree). Once loaded, it will print the degradation statistics for each attack type. In the paper, the median was used. 

# Notes
Here are a few useful tips:
1. The plots obtained from these scripts may not look appear exactly like those used in the paper - those were prepared using a different plotting software, but the plots were generated from the same data produced by these scripts (hence the option to save the raw plot data). 
2. For experiments requiring different classifier settings (specifically the sensitivity analyses), it is useful to: edit the parameters in the Hypothesis 3 scripts, and adjust the classifier code e.g when changing the LTS base length from 0.1 to 0.05, the classifier code LTS5 could be used. Next, execute the `run_tests.py` script, which will produce results in the result directory all tagged LTS5-xxxxxx. The CSV result file may then be generated using this new classifier code, allowing for easy side by side analysis as required.
