# Park & Drive Potential

## Installation

Install [uv](https://docs.astral.sh/uv/) and then
set up the python project like this:

```bash
uv sync # set up project + dependencies
uv run pytest # run unit tests to see if installation was successful
```

## Usage

1. Prepare a `GeoPackage` containing layers for
   - the MATSim network (line geometry)
   - park & drive stations to be checked (point geometry)
   - cities & large towns (point geometry) for visualization purposes
   - barriers (any geometry) for limiting buffers around the park & drive stations

2. Extract all trips from a calibrated MATSim population file to a `CSV` file
   - Must include info about subtours
   - Must include routes (as a list of link ids) for all car trips
   - Header: `personId,age,sex,isEmployed,fullTimeWork,highEducation,highIncome,subtourNr,subtourLen,tripNr,mode,departureSecondsOfDay,arrivalSecondsOfDay,nextActivityStartSecondsOfDay,activityChain,links,inEducation,retired` (subtourNr and tripNr must be 1-based)
   - E.g. adjust this [exemplary script](misc/PopulationExtractSubTours.java)


3. Run the analysis:
   ```bash
   uv run main.py
   ```

## Methodology & Results

The inner workings and outputs/results of the tool will be described in detail in
*Quantifying Park & Drive Potential: A Quick Location Planning Tool*,
an upcoming paper submitted at Transport Research Arena (TRA) 2026.

# Acknowledgements

This research was funded by the Austrian Research Promotion Agency (FFG)
under grant [4906625, project INTRO (Integrierte Mobilitätsknoten)](https://projekte.ffg.at/projekt/4906625).
