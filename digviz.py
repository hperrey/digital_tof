#!/usr/bin/env python3
"""

Plot tool for csv data files recorded with CAEN digitizers running DPP-PH firmware

"""

import sys
import os
import argparse
import matplotlib
matplotlib.use('Qt5Agg')  # nice, but issue with interactive use e.g. in
                          # Jupyter; see
                          # http://matplotlib.org/faq/usage_faq.html#what-is-a-backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

import h5py
from enum import Enum

import colorer # colored log output

class DataType(Enum):
    # define the supported data formats (that need individual parsing)
    DPPQDC = 1
    DPPQDC_EXT = 2
    STANDARD = 3
    DPPQDC_MIXED = 257
    DPPQDC_EXT_MIXED = 258

def load_csv(file_name):
    '''
    Loads data from a CSV file. Formats currently supported:
    - DPP-QDC data (XX740-family)
    '''
    return pd.read_csv(file_name, sep=" ", header=None, names=['ts','dig','ch','ph'])

def load_h5(file_name):
    """
    Loads data stored with JADAQ from an HDF5 file. Supports so far:
    - Standard wave function (recorded with XX751)
    """
    log = init_logging('digviz')  # set up logging

    f = h5py.File(file_name, 'r')
    # TODO check version of data format in file and give out warning if newer than expected

    all_digi = list(f.keys())
    log.info("File {} contains data from digitizers: {}".format(file_name, ",".join(all_digi)))

    dfs = []

    for digi in all_digi:
        # TODO catch exception should data format not be known
        dformat = DataType(f[digi].attrs.get('JADAQ_DATA_TYPE'))
        log.info(f"Data format for digitizer {digi}: {dformat}")

        if dformat == DataType.STANDARD:
            evtform = np.dtype([('evtno', np.uint32), ('ts', np.uint32), ('ch', np.uint8),
                                ('samples', np.object)])

            # data length calculation
            nevts = sum([len(f[digi][g]) for g in f[digi].keys()])
            # each event consists of several active channels; get number from channel mask of first event
            first_data = list(f[digi].keys())[0]
            first_event = f[digi][first_data][0]
            chmask = first_event[1]
            nch = bin(chmask).count('1') # count active channels in mask
            # compose list of active channels from channel mask
            channels = [position for position, bit in enumerate([(chmask >> bit) & 1 for bit in range(8)]) if bit]
            # adjust number of events accordingly
            nevts *= nch

        nblocks = len(list(f[digi].keys()))
        log.info("Found a total of {} samples (or events) for {} active channels stored in {} data blocks".format(nevts, len(channels), nblocks))

        # reserve memory for numpy data array holding events
        data = np.zeros(nevts, dtype=evtform)

        # loop over all events in all data blocks stored for this digitizer
        for idx, evt in enumerate([e for dblock in list(f[digi].keys()) for e in f[digi][dblock]]):
                if dformat == DataType.STANDARD:
                    # split the combined samples for all channels and create single events
                    for i,s in enumerate(np.split(evt[4],nch)):
                        data[idx*nch+i] = np.array((evt[2], evt[0], np.uint8(channels[i]), s), dtype=evtform)
        df = pd.DataFrame(data)
        # add an identifier column for the digitizer model+serial
        df["digitizer"] = str(digi) # TODO str wastes storage space here; keep a uint8 digitizer index and a map to the name instead?
        dfs.append(df)

    # concatenate results (if needed)
    if len(dfs) == 1:
        return dfs[0]
    else:
        return pd.concat(dfs)

def load_df(file_name):
    if str(file_name).endswith('.csv'):
        return load_csv(file_name)
    elif str(file_name).endswith('.h5'):
        return load_h5(file_name)
    else:
        print(f"Unknown file ending, only supported types are 'csv' and 'h5': {file_name}")
        sys.exit(2)

def calculate_time_difference(dataframe):
    """
    Adds a column 'delta_ts' with time difference between neighbouring
    events. Potential overflows are reported as warnings in the log.
    """
    log = init_logging('digviz')  # set up logging
    dataframe['delta_ts'] = (dataframe['ts'] - dataframe['ts'].shift()).fillna(0)
    for index, row in dataframe[dataframe['delta_ts'] < 0].iterrows():
        log.warning(
            "Found potential overflow: timestamp difference between event (index {}) with timestamp {} and the next one is: {}".
            format(row.name, row['ts'], row['evtno'], row['delta_ts']))
    dataframe.loc[
        dataframe.delta_ts < 0,
            'delta_ts'] += 4294967295  # add 32-bit max value where ever there was a overflow

def plot_ch(measurements, labels=None):
    fig, ax = plt.subplots()
    for idx, df in enumerate(measurements):
        l = "file {}".format(idx)
        try:
            l = labels[idx]
        except:
            pass
        df['ch'].plot.hist(bins=64, label=l)
    plt.xlabel("channel #")
    plt.legend()
    return fig, ax

def plot_ph(measurements, query=None, labels=None):
    fig, ax = plt.subplots()
    for idx, df in enumerate(measurements):
        l = "file {}".format(idx)
        try:
            l = labels[idx]
        except:
            pass
        if query is not None:
            df.query(query)['ph'].plot.hist(alpha=.65, bins=100, label=l)
        else:
            df['ph'].plot.hist(alpha=.65, bins=100, label=l)
    plt.xlabel("PH")
    plt.title(query)
    plt.legend()
    return fig, ax


def persistantTracePlot(samples, gridsize=250):
    ''' show "persistent" trace picture '''
    fig, ax = plt.subplots()
    time = np.concatenate([range(len(s)) for s in samples])
    ampl = np.concatenate([s for s in samples])
    plt.xlabel('time')
    plt.ylabel('ampl')
    plt.hexbin(time, ampl, bins='log', gridsize=gridsize)
    plt.title("persistent trace in 2D hex binning")
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    # adjust axis range
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,1.1*y1,1.1*y2))
    return fig, ax


def plot_persistantTrace(measurements, query=None, labels=None):
    for idx, df in enumerate(measurements):
        l = "file {}".format(idx)
        try:
            l = labels[idx]
        except:
            pass
        if not 'samples' in df.columns:
            continue
        if query is not None:
            samples = df.query(query)['samples']
        else:
            samples = df['samples']
        persistantTracePlot(samples)
        plt.title(f"persistent trace for {l}")

def plot_samples(measurements, query=None, labels=None):
    fig, ax = plt.subplots()
    # set up colors
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0, 1, len(measurements))]

    for idx, df in enumerate(measurements):
        l = "file {}".format(idx)
        try:
            l = labels[idx]
        except:
            pass
        if not 'samples' in df.columns:
            continue
        if query is not None:
            samples = df.query(query)['samples']
        else:
            samples = df['samples']
        for sidx, s in enumerate(samples):
            ax.plot(range(len(s)), s, color=colors[idx],
                    label=f"{l}" if sidx==0 else "_nolegend_")
        plt.legend()
    return fig, ax

def plot_samples_per_channel(measurements, query=None, labels=None):
    fig, ax = plt.subplots()

    for idx, df in enumerate(measurements):
        l = "file {}".format(idx)
        try:
            l = labels[idx]
        except:
            pass
        if not 'samples' in df.columns:
            continue
        if query is not None:
            gb = df.query(query).groupby(['ch'])
        else:
            gb = df.groupby(['ch'])
        # set up colors
        cmap = plt.get_cmap('jet')
        colors = [cmap(i) for i in np.linspace(0, 1, len(gb))]
        for name, grouped_df in gb:
            if query is not None:
                samples = grouped_df.query(query)['samples']
            else:
                samples = grouped_df['samples']
            for sidx, s in enumerate(samples):
                ax.plot(range(len(s)), s, color=colors[name],
                        label=f"{l} ch {name}" if sidx==0 else "_nolegend_")
        plt.legend()
    return fig, ax


def plot_ph_per_channel(measurements, query=None, labels=None):
    fig, ax = plt.subplots()
    for idx, df in enumerate(measurements):
        l = "file {}".format(idx)
        try:
            l = labels[idx]
        except:
            pass
        if query is not None:
            gb = df.query(query).groupby(['ch'])
            for name, grouped_df in gb:
                grouped_df['ph'].plot.hist(alpha=.65, bins=100, label="{} ch {}".format(l,name))
        else:
            gb = df.groupby(['ch'])
            for name, grouped_df in gb:
                grouped_df['ph'].plot.hist(alpha=.65, bins=100, label="{} ch {}".format(l,name))
    plt.xlabel("PH")
    plt.title(query)
    plt.legend()
    return fig, ax


def init_logging(logger, log_level = "INFO"):
    """ initializes logging """
    log = logging.getLogger(logger)  # get reference to logger
    # test if we have set up the logger before
    if not len(log.handlers):
        # perform setup by adding handler:
        formatter = logging.Formatter('%(asctime)s %(name)s(%(levelname)s): %(message)s',"%H:%M:%S")
        handler_stream = logging.StreamHandler()
        handler_stream.setFormatter(formatter)
        log.addHandler(handler_stream)
        # using this decorator, we can count the number of error messages
        class callcounted(object):
            """Decorator to determine number of calls for a method"""
            def __init__(self,method):
                self.method=method
                self.counter=0
            def __call__(self,*args,**kwargs):
                self.counter+=1
                return self.method(*args,**kwargs)
        log.error=callcounted(log.error)
        # set the logging level
        numeric_level = getattr(logging, log_level,
                                None)  # default: INFO messages and above
        log.setLevel(numeric_level)
    return log


if __name__ == "__main__":
    log = init_logging('mapmt_plotter')

    # command line argument parsing
    argv = sys.argv
    progName = os.path.basename(argv.pop(0))
    parser = argparse.ArgumentParser(
        prog=progName,
        description=
        "Visualize CSV data files recorded with CAEN digitizers running DPP-PH mode" # TODO: check accuracy some time later again
    )
    parser.add_argument(
        "-l",
        "--log-level",
        default="info",
        help=
        "Sets the verbosity of log messages where LEVEL is either debug, info, warning or error",
        metavar="LEVEL")
    parser.add_argument(
        "-i",
        "--interactive",
        action='store_true',
        help=
        "Drop into an interactive IPython shell instead of showing default plots"
    )
    parser.add_argument(
        "file",
        nargs='+',
        help=
        "The csv data file(s) to be processed; additional info STRING to be included in the plot legend can be added by specifiying FILE:STRING"
    )

    # parse the arguments
    args = parser.parse_args(argv)
    # set the logging level
    numeric_level = getattr(logging, "DEBUG",
                            None)  # set default
    if args.log_level:
        # Convert log level to upper case to allow the user to specify --log-level=DEBUG or --log-level=debug
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            log.error('Invalid log level: %s' % args.log_level)
            sys.exit(2)
    log.setLevel(numeric_level)

    # parse file names and extract additionally provided info
    fileNames = []
    fileDescr = []
    for thisFile in args.file:
        s = thisFile.strip().split(':', 1)  # try to split the string
        if (len(s) == 1):
            # didn't work, only have one entry; use file name instead
            fileNames.append(s[0])
            fileDescr.append(s[0])
        else:
            fileNames.append(s[0])
            fileDescr.append(s[1])

    log.debug("Command line arguments used: %s ", args)
    log.debug("Libraries loaded:")
    log.debug("   - Matplotlib version {}".format(matplotlib.__version__))
    log.debug("   - Pandas version {}".format(pd.__version__))
    log.debug("   - Numpy version {}".format(np.__version__))

    msrmts = [] # list to hold all measurements
    for idx, f in enumerate(fileNames):
        df = load_df(f)
        if df is None:
            log.error("Error encountered, skipping file {}".format(f))
            continue
        msrmts.append(df)
        mem_used = df.memory_usage(deep=True).sum()
        log.info("Approximately {} MB of memory used for data loaded from {}".format(round(mem_used/(1024*1024),2),f))

    if (args.interactive):
        print(" Interactive IPython shell ")
        print(" ========================= ")
        print(" Quick command usage:")
        print("  - 'who' or 'whos' to see all (locally) defined variables")
        print("  - if the plots are shown only as black area, run '%gui qt'")
        print("  - to make cmd prompt usable while plots are shown use 'plt.ion()' for interactive mode")
        print("    or run 'plt.show(block=False)' to show them")
        import IPython
        IPython.embed()
    else:
        plot_ch(msrmts, labels=fileDescr)
        plot_samples(msrmts, labels=fileDescr)
        plot_persistantTrace(msrmts, labels=fileDescr)

        plt.show(block=False)  # block to allow "hit enter to close"
        plt.pause(0.001)  # <-------
        input("<Hit Enter To Close>")
        plt.close('all')

    if log.error.counter>0:
        log.warning("There were "+str(log.error.counter)+" error messages reported")
