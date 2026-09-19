## Dataset

The public benchmark datasets used in this study (PeMS03, PeMS04, PeMS07, and PeMS08) are hosted on Baidu Netdisk:

Link: https://pan.baidu.com/s/1EW1mklX7uQ5rKZhcQmvRtA?pwd=qv1p  
Password: qv1p

Download and place the extracted `data` folder under the project root directory:

`MS-STVMamba/data/`

# BT-MSTF Dataset Construction

BT-MSTF is a multi-source urban traffic forecasting dataset constructed in
Baotou, Inner Mongolia, China. It integrates traffic-flow observations,
road-network information, point-of-interest (POI) features, and weather
conditions.

The main data-collection and preprocessing procedures are described below.

## 1. Traffic Flow Data

The traffic-flow data were obtained from the traffic management authority of
Baotou, China. The original traffic observations were recorded at a temporal
resolution of 5 minutes.

During dataset construction, traffic monitoring nodes located in the central
urban area were first selected. Nodes with more than 10% missing traffic-flow
observations were removed.

For the retained nodes, the remaining missing observations were processed using
causal forward filling, where each missing value was replaced by the most
recent valid observation preceding it.

The processed traffic-flow data are organized as a three-dimensional array:

(time steps, nodes, features)

After preprocessing, 290 traffic monitoring nodes were retained.

## 2. Road Network Data

The road-network data were constructed based on the spatial locations and road
connectivity of the retained traffic monitoring nodes.

The monitoring nodes were visualized on a map, and the connections between
nodes were identified according to the corresponding road-network structure.

Each road connection is represented by:

[from, to, distance]

where `from` and `to` denote the identifiers of the connected traffic
monitoring nodes, and `distance` represents the distance between the two nodes.

The resulting road network contains 225 connections.

## 3. POI Data

POI information was collected within a 600-m radius around each traffic
monitoring node.

For each node, the numbers of POIs belonging to different functional
categories were counted to construct node-level semantic features.

The POI data contain the following fields:

[device, hospital, education, retail, residence, recreation, industrial,
office_facilities, public_institution, transportation]

Here, `device` denotes the identifier of the corresponding traffic monitoring
node. The remaining fields record the numbers of POIs belonging to the
corresponding functional categories within a 600-m radius of that node.

## 4. Weather Data

The original weather observations were obtained from an online meteorological
data source at an hourly temporal resolution.

The original weather data contain:

[date, weather_condition]

Different weather conditions were first converted into numerical weather codes
according to the encoding scheme described in the manuscript.

Because the traffic-flow observations are sampled every 5 minutes, the encoded
hourly weather sequence was subsequently expanded to the traffic-flow temporal
resolution. Specifically, each hourly weather code was repeated for 12
consecutive 5-min time steps corresponding to the same hour.

This procedure ensures temporal alignment between the weather and traffic-flow
sequences.

## 5. Multi-source Data Alignment

The four data sources were aligned using the traffic monitoring nodes and
traffic timestamps as the common spatial and temporal references.

- Traffic-flow data provide the 5-min temporal sequence for each monitoring
  node.
- Road-network data describe the spatial connectivity among traffic nodes.
- POI data describe the surrounding functional characteristics of each node
  within a 600-m radius.
- Weather conditions are first numerically encoded and then expanded from the
  hourly resolution to 5-min intervals to match the traffic-flow timestamps.

The resulting BT-MSTF dataset therefore contains spatially and temporally
aligned traffic-flow, road-network, POI, and weather information for
multi-source traffic forecasting.

## 6. Data Organization

The processed datasets follow the directory structure below:

data/
├── btmstf/
│   ├── btmstf.csv
│   ├── btmstf.npz
│   ├── poi.csv
│   └── weather.csv
├── PEMS03/
│   ├── PEMS03.csv
│   └── PEMS03.npz
├── PEMS04/
│   ├── PEMS04.csv
│   └── PEMS04.npz
├── PEMS07/
│   ├── PEMS07.csv
│   └── PEMS07.npz
└── PEMS08/
    ├── PEMS08.csv
    └── PEMS08.npz

For BT-MSTF:

- `btmstf.npz` contains the processed traffic-flow sequence used as the main
  temporal input.
- `btmstf.csv` stores the road-network connections in the form
  `[from, to, distance]`.
- `poi.csv` contains the node identifier and the corresponding POI-category
  counts.
- `weather.csv` contains the temporally aligned weather information used by
  the multi-source model.

The PeMS folders contain the corresponding public traffic-flow data and
road-network information used for the traffic-only benchmark experiments.

## 7. Data Availability

The original BT-MSTF dataset cannot currently be publicly released due to
data-sharing restrictions imposed by the relevant data management authority.

To improve transparency and reproducibility, this repository provides the
MS-STVMamba implementation together with the dataset-construction,
preprocessing, data-organization, and multi-source alignment procedures
described above. Researchers with similarly structured source data can use
these procedures to reproduce the BT-MSTF data-processing and modeling
pipeline.
