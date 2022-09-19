# Wikinlg
# Project: Filtering Distant Supervision Data

## About this project
In this project, we apply various filtering techniques on RDF/NL data created using distant supervision and explore impact on RDF-to-Text generation. We fine tune pretained models on top of filtered data from KELM and TekGen which covers more domains and not limited to WebNLG. The filtering of TekGen and KELM is done by finding semantic similarity between RDF triples and reference text.

## Installation
Note: Following code has been implemented in Python3

1. cd to the directory where ```requirements.txt``` is located;

2. run: `pip install -r requirements.txt` .

## Dataset
TekGen and KELM can be downloaded from  https://github.com/google-research-datasets/KELM-corpus

## How to execute the code
In order to execute the code from this repository and get the results, the following commands need to be run:

`cd Filter`

`python calculatefilterscore.py --input-path <path including filename> --save-path <directory to save filter score> --save-filename <filename to save parent filter score>`

Each line is an example as a json object with three fields:

1. triples: A list of triples of the form (subject, relation, object). eg. (Person X, award received, Award Y). If the triple has a subproperty, then it is quadruple instead. eg. (Person X, Award Y, received on, Date Z). 

2. serialized triples: triples concatenated together as used for input to T5. The format is "&lt;subject&gt; &lt;relation&gt; &lt;object&gt;" where some subjects have multiple relations, e.g. "&lt;subject&gt; &lt;relation1&gt; &lt;object1&gt; &lt;relation2&gt; &lt;object2&gt; &lt;relation3&gt; &lt;object3&gt;".

3. Text: The generated natural language sentence for the triples.


final filename with `parent<filename>.jsonl` with be stored

`parent<filename>.jsonl` contains original data along with sim_score, gleu_score, bleu_score , parent_score , average_score.

 ### Prepare test data
 
 `cd simtestdata`
 
 `python data_to_csv.py --dataset-path <path of file generated in the previous step> --save-path <directory to save filter data and webnlg data>`
  
  This script converts data to csv. Two files are created noisydatacsv and webnlg.csv
  
  `python split_noisy_val_dataset.py --noisy-dataset-path <path of noisydata.csv generated in the previous step> --webnlg-dataset-path <path to webnlg.csv>    --save-path <directory to save Wikinlg train and test data > --language-code <language of the data>`
  
## Fine Tune Model

 We use pretrained T5 base model provided by hugging face to finetune on the filtered data.
 
 `cd models`
 
 `python train_t5_gen.py --data-path <train+dev data> --language "english" --model-name "t5-base" --save-dir "models/" --log-dir "log/" --batch_size "32" --epochs "5"`

Input data must be in csv format with triples and text as columns. Triples must be a list of list of triples per instance. 

## Evaluation 
For evaluation , please refer https://github.com/WebNLG/GenerationEval repository.
 
 
