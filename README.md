### Cased Borehole Cement Evaluation

This is a implementation of method proposed by [Erlend Magnus Viggen's article](https://www.sciencedirect.com/science/article/pii/S0920410520306100)

It consists of a multimodal CNN trained using integrity log data from the public [Volve database](https://www.equinor.com/energy/volve-data-sharing), provided by Equinor.

In the original article, the results are based on private data. However, in this repo, you will find results based solely on data from three offshore oil wells available in the public database.

### Work In Progress

All .dlis files used to train the model were extracted from the “11. INTEGRITY LOGS” folder in the Volve dataset and were stored in the /dataset/dlis_data folder.

The dataset contains specialist evaluation about cement bond integrity and hydraulic isolation on .PDF reports at Volve's folder "11. INTEGRITY LOGS". Three of the four available wells have notes regarding hydraulic isolation: F-9, F-11B, and F-12.

Except for well F-11B, which has only two labels (“yes” and ‘no’), the others have three labels (“High,” “Medium,” and “Low”). To standardize the labels in the database, the labels for wells F-9 and F-12 were converted as follows: [“Medium,” “High”] = “yes,” "Low" = “no.”

Only wells F-9 and F-11B have labels indicating cement quality: “Free Pipe,” “Poor,” “Poor to Moderate,” “Moderate,” “Moderate to Good,” and “Good.” Well F-12 is labeled differently, so someone who is not a petrophysicist might not interpret it correctly. Therefore, these experiments will not include well F-12.

The labels extracted from the PDF reports are available at /dataset/labels

### References:

- Viggen, Erlend Magnus, et al. "Automatic interpretation of cement evaluation logs from cased boreholes using supervised deep neural networks." Journal of Petroleum Science and Engineering 195 (2020): 107539.

- Viggen, Erlend Magnus, Erlend Hårstad, and Jørgen Kvalsvik. "Getting started with acoustic well log data using the dlisio Python library on the Volve Data Village dataset." 43rd Scandinavian Symposium on Physical Acoustics, Geilo, Norway. 2020.
