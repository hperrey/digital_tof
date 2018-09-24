#!/usr/bin/env python3
"""

Plot tool for csv data files recorded with CAEN digitizers running DPP-PH firmware

"""

import sys
import os
import argparse
import matplotlib
from math import ceil as round_up
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
                    #evt[2]=channelnumber, evt[0]=timestamp, s=samples, i=nch=sum of 1 in binary rep of channel mask
                    for i,s in enumerate(np.split(evt[4],nch)):
                        data[idx*nch+i] = tuple([evt[2], evt[0], np.uint8(channels[i]), s])
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
            'delta_ts'] += 2147483647  # add half of 32-bit max value where ever there was a overflow

def cfd(samples, peak, frac):
    refpoint = 0
    for i in range(0,len(samples)):
        if samples[i]>= frac*peak:
            refpoint = i
            break
    return refpoint

def find_edges(samples, point):
    edges = [0,0]
    for i in range(point, 0, -1):
        if samples[i]<2:
            edges[0]=i
            break
    for i in range(point, len(samples)):
        if samples[i]<2:
            edges[1]=i
            break
    return edges

def process_frame(dataframe, threshold=[20,20,20,20,20,20,20,20], frac=0.5):
    """Process a dataframe to extract more features: area, longate and shortgate area, edges, height,
    alignment/refpoint, and to correct the timestamp.
    Parameters: dataframe: the dataframe to be processed, Threshold: 8 element list specifying pulse heightthreshold for each channel,
    frac: the constant fraction used in the cfd, sg and lg: the fraction of the pulse to be used in long and short gate integration."""
    event_index = 0
    nTimeResets = 0
    length = len(dataframe)
    pr_samples = [0]*length
    left = np.array([0]*length, dtype=np.int16)
    right = np.array([0]*length, dtype=np.int16)
    pr_ts = np.array([0]*length,dtype=np.uint64)
    pr_evtno = np.array([0]*length, dtype=np.uint32)
    pr_ch = np.array([0]*length,dtype=np.uint8)
    pr_digitizer = np.array(['']*length, dtype=str)
    refpoint = np.array([0]*length, dtype=np.int16)
    height = np.array([0]*length, dtype=np.int16)
    area = np.array([0]*length, dtype=np.int16)
    longgate = np.array([0]*length, dtype=np.int16)
    lg=0.9
    shortgate = np.array([0]*length, dtype=np.int16)
    sg=0.20
    for i in range(0, length):
        if i%10 == 0:
            p = round_up(100*i/length)
            sys.stdout.write("\rprocessing dataframe %d%%" % p)
            sys.stdout.flush()

        #shift baseline using first 20 sample points.
        pr_samples[i] = dataframe.samples[i].astype(np.int16)
        pr_samples[i] -= np.int((sum(pr_samples[i][0:20])/20))

        #keep track of the timestamp resets
        if i > 0:
            if dataframe.ts[i] < dataframe.ts[i-1]:
                nTimeResets += 1
        #apply threshold and make all pulses positive
        peak_index = np.argmax(np.absolute(pr_samples[i]))
        if abs(pr_samples[i][peak_index]) >= threshold[dataframe.ch[i]]:
            if pr_samples[i][peak_index] < 0:
                pr_samples[i] *= -1
            left[i], right[i] = find_edges(pr_samples[i], peak_index)
            #check if events are fully contained in event window and of a certain width
            if right[i]-left[i]>5:
                left[event_index] = left[i]
                right[event_index] = right[i]
                #We have an acceptable event, so we save it
                pr_samples[event_index] = pr_samples[i]
                pr_ts[event_index] = (dataframe.ts[i]+nTimeResets*2147483647)
                pr_evtno[event_index] = dataframe.evtno[i]
                pr_ch[event_index] = dataframe.ch[i]
                pr_digitizer[event_index] = dataframe.digitizer[i]
                height[event_index]=pr_samples[event_index][peak_index]
                refpoint[event_index] = cfd(pr_samples[event_index], height[event_index],frac)
                area[event_index] = np.trapz(pr_samples[event_index][left[event_index]:right[event_index]])
                width = right[event_index]-left[event_index]
                longgate[event_index] = np.trapz(pr_samples[event_index][left[event_index]:left[event_index]+int(lg*width)])
                shortgate[event_index] = np.trapz(pr_samples[event_index][left[event_index]:left[event_index]+int(sg*width)])
                #longgate[event_index] = np.trapz(pr_samples[event_index][edges[event_index][0]:edges[event_index][0]+int(lg*width)])
                #shortgate[event_index] = np.trapz(pr_samples[event_index][edges[event_index][0]:edges[event_index][0]+int(sg*width)])
                event_index += 1
    #Some event were sorted out, so we drop the redundant elements in the lists
    pr_samples = pr_samples[0:event_index]
    pr_ts = pr_ts[0:event_index]
    pr_evtno = pr_evtno[0:event_index]
    pr_ch = pr_ch[0:event_index]
    pr_digitizer = pr_digitizer[0:event_index]
    refpoint = refpoint[0:event_index]
    left = left[0:event_index]
    right = right[0:event_index]
    height = height[0:event_index]
    area = area[0:event_index]
    longgate = longgate[0:event_index]
    shortgate = shortgate[0:event_index]
    return pd.DataFrame({"samples":pr_samples,
                         "ts":pr_ts,
                         "evtno":pr_evtno,
                         "ch":pr_ch,
                         "digitizer":pr_digitizer,
                         "refpoint":refpoint,
                         "left":left,
                         "right":right,
                         "height":height,
                         "area":area,
                         "longgate":longgate,
                         "shortgate":shortgate})

def tof_spectrum(frame, fac=8, tolerance = 1000):
    #Needs to be generalised to different channels!
    counter=0
    ne213 = fac*frame.query("ch==0").ts.values+frame.query("ch==0").refpoint.values
    yap = fac*frame.query("ch==1").ts.values+frame.query("ch==1").refpoint.values
    ymin = 0
    tof_hist = np.histogram([], 2*tolerance, range=(-tolerance, tolerance))
    for ne in range(0, len(ne213)):
        counter += 1
        k = round_up(100*counter/len(ne213))
        sys.stdout.write("\rGenerating tof spectrum %d%%" % k)
        sys.stdout.flush()
        for y in range(ymin,len(yap)):
            print(ne, ':', y)
            Delta=ne213[ne]-yap[y]
            if Delta > tolerance:
                ymin = y
            if -tolerance < Delta <tolerance:
                tof_hist[0][tolerance+int(Delta)] += 1
            elif Delta < -tolerance:
                break
    return tof_hist




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
    log = init_logging('digiviz')

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
