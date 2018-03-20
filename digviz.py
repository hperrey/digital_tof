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

import colorer # colored log output


def load_df(file_name):
    return pd.read_csv(file_name, sep=" ", header=None, names=['ts','dig','ch','ph'])

def calculate_time_difference(dataframe):
    """
    Adds a column 'delta_ts' with time difference between neighbouring
    events. Potential overflows are reported as warnings in the log.
    """
    log = logging.getLogger('digviz')  # set up logging
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


if __name__ == "__main__":
    log = logging.getLogger('digviz')  # set up logging
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
        plot_ch(msrmts, fileDescr)

        plt.show(block=False)  # block to allow "hit enter to close"
        plt.pause(0.001)  # <-------
        input("<Hit Enter To Close>")
        plt.close('all')

    if log.error.counter>0:
        log.warning("There were "+str(log.error.counter)+" error messages reported")
